"""
iTransformer — 含纯净版 Model + GeoHCAN Geometric MoE Enhancement

Paper: https://arxiv.org/abs/2310.06625

两个类:
    Model    — 纯净版 iTransformer (--no_geohcan 使用)
    Backbone — GeoHCAN 几何 MoE 增强模式 (默认使用)

GeoHCAN 几何 MoE 增强 (Geometric MoE Enhancement):
    核心思路: 利用质心路由空间对变量进行分类, 每个质心代表一种变量模式,
    通过 Correction MLP 为不同模式的变量学习不同的修正策略 (类似 MoE).

    关键设计:
    1. 路由空间冻结 (K-Means init 后): 质心 + 投影层不更新, 防止崩溃
    2. Correction MLP 零初始化: 起步 correction = 0, 模型完全等价原生 iTransformer
    3. 可学习全局门控: sigmoid(gate_alpha), 初始 ≈ 0.02, 逐步开放
    4. 无 gradient detach: encoder 接收几何反馈 → 学习产生更好的特征

    数据流:
        x_enc [B, T, N] → RevIN → Embed → Encoder → enc_out [B, N_tok, 512]
        → GeoHCAN MoE Enhancement (NO detach):
            routing: proj(512→d_proj) → cosine sim → soft_assign [B, N_tok, K]
            context: soft_assign @ centroids → centroid_ctx [B, N_tok, d_proj]
            correction: MLP([x_norm || centroid_ctx]) → [B, N_tok, 512]
            out = enc_out + sigmoid(gate_alpha) * correction
        → out → Projection(512→P) → [B, P, N]
        → RevIN Denorm → output
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


# =====================================================================
#  iTransformer 专用: 带几何注意力偏置的 Attention/Encoder 子类
#  (不修改 layers/ 下的共享文件, 其他模型不受影响)
# =====================================================================
class GeoFullAttention(FullAttention):
    """FullAttention + 几何注意力偏置 (仅 iTransformer 使用)."""
    def forward(self, queries, keys, values, attn_mask, attn_bias=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # iTransformer: mask_flag=False, 无需 causal mask
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, float('-inf'))
        # 几何注意力偏置: 在 softmax 前注入变量间相似度
        logits = scale * scores
        if attn_bias is not None:
            logits = logits + attn_bias  # [B, 1, N, N] broadcast across H heads
        A = self.dropout(torch.softmax(logits, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


class GeoAttentionLayer(AttentionLayer):
    """AttentionLayer — 透传 attn_bias 给 GeoFullAttention."""
    def forward(self, queries, keys, values, attn_mask, attn_bias=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, attn_bias=attn_bias
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class GeoEncoderLayer(EncoderLayer):
    """EncoderLayer — 透传 attn_bias."""
    def forward(self, x, attn_mask=None, attn_bias=None):
        new_x, attn = self.attention(
            x, x, x, attn_mask=attn_mask, attn_bias=attn_bias
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class GeoEncoder(Encoder):
    """Encoder — 透传 attn_bias 到每一层."""
    def forward(self, x, attn_mask=None, attn_bias=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, attn_bias=attn_bias)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# =====================================================================
#  VariateGeoHCAN — 几何 MoE 增强器 (Geometric MoE Enhancement)
# =====================================================================
class VariateGeoHCAN(nn.Module):
    """
    Geometric MoE Enhancement for iTransformer variate tokens.

    核心升级 (相比旧版 Uncertainty Isolation):
      - 旧版: reliability gate + safe_repr fallback → 只能抑制, 不能增强
      - 新版: centroid-conditioned correction MLP → 为不同模式的变量学习不同修正

    架构:
      1. 几何路由 (frozen after K-Means):
         proj(d_model→d_proj) → cosine sim → soft_assign [B, N, K]
      2. 质心条件修正 MLP (zero-initialized):
         MLP([x_norm || centroid_ctx]) → correction [B, N, d_model]
      3. 门控残差连接:
         out = x + sigmoid(gate_alpha) * correction

    安全保证:
      - 路由空间冻结 → 质心不崩溃, 投影不漂移
      - Correction MLP 零初始化 → 初始 correction = 0 → 完全等价原生 iTransformer
      - Gate 初始 sigmoid(-4) ≈ 0.018 → 修正从极小值逐步增长
      - 无 gradient detach → encoder 获得几何反馈 → 产生更优特征

    兼容接口:
      - self.proj, self.proj_norm, self.centroids, self.n_centroids → K-Means init
      - forward 返回 (out, u_std, T_mean, attn, ortho, geo, latent) → 统一接口
    """

    def __init__(self, d_model, d_proj=32, n_centroids=6,
                 gate_power=2, detach_temp=True, use_proj_ln=True):
        super(VariateGeoHCAN, self).__init__()
        self.d_model = d_model
        self.d_proj = d_proj
        self.n_centroids = n_centroids

        # ===== 几何路由空间 (K-Means 初始化后冻结) =====
        self.proj = nn.Linear(d_model, d_proj)
        self.proj_norm = nn.LayerNorm(d_proj) if use_proj_ln else nn.Identity()

        # ===== 质心 (代表 K 种变量模式) =====
        self.centroids = nn.Parameter(torch.empty(n_centroids, d_proj))
        nn.init.uniform_(self.centroids, -3.0, 3.0)

        # ===== Reliability Gate (诊断 + 兼容, K-Means 后校准) =====
        self.rel_slope = nn.Parameter(torch.tensor(10.0))
        self.rel_threshold = nn.Parameter(torch.tensor(0.5))

        # ===== 质心条件修正 MLP (带 Dropout 防过拟合) =====
        # 输入: [x_norm (d_model) || centroid_ctx (d_proj)]
        # 输出: correction (d_model)
        corr_hidden = max(d_model // 16, 32)  # 32 for d_model=512 (更小, 防过拟合)
        self.correction_in = nn.Linear(d_model + d_proj, corr_hidden)
        self.correction_act = nn.GELU()
        self.correction_drop = nn.Dropout(0.15)  # 防止过拟合
        self.correction_out = nn.Linear(corr_hidden, d_model)
        # 零初始化: correction = 0 at start → 模型完全等价原生 iTransformer
        nn.init.zeros_(self.correction_out.weight)
        nn.init.zeros_(self.correction_out.bias)

        # ===== 全局门控 =====
        # sigmoid(0) = 0.5 → correction MLP 从第一步就拿到 50% 梯度
        # 但因为 correction_out 零初始化, 初始 output = x + 0.5*0 = x (安全)
        self.gate_alpha = nn.Parameter(torch.tensor(0.0))

        # ===== 路由温度 (可学习, 控制 soft-assign 锐利度) =====
        # exp(1.6) ≈ 5.0 → 中等锐利度
        self.routing_log_temp = nn.Parameter(torch.tensor(1.6))

        # ===== Evidence Engine (诊断用, 不影响主路径) =====
        self.scale_evidence = nn.Parameter(torch.tensor(5.0))
        self.alpha_dir = 5.0
        self.softplus = nn.Softplus()

        # ===== Debug 计步器 =====
        self._fwd_count = 0

    def forward(self, x, y_true=None):
        """
        Args:
            x: [B, N, d_model]  — N 个异质变量 token (encoder 输出)
        Returns:
            out:         [B, N, d_model]  — 增强后的表征
            u_std:       [B, N, 1]        — DST 不确定度 (诊断用)
            T_mean:      scalar 0          — 占位
            attn:        None              — 占位
            ortho_loss:  scalar            — 质心正交损失
            geo_loss:    scalar 0          — 占位
            latent_loss: scalar            — 负载均衡损失 (鼓励质心均匀使用)
        """
        B, N, D = x.shape

        # ===== Internal RevIN (跨变量归一化, 消除量纲差异) =====
        rev_mean = x.mean(dim=1, keepdim=True)      # [B, 1, d_model]
        rev_std = x.std(dim=1, keepdim=True) + 1e-5  # [B, 1, d_model]
        x_norm = (x - rev_mean) / rev_std

        # ===== 几何路由: 投影 → 余弦相似度 → 软分配 =====
        Q = self.proj_norm(self.proj(x_norm))            # [B, N, d_proj]
        Q_n = F.normalize(Q, p=2, dim=-1)
        K_n = F.normalize(self.centroids, p=2, dim=-1)   # [K, d_proj]
        sim_cos = torch.matmul(Q_n, K_n.T)               # [B, N, K]

        # Reliability (诊断用, 兼容 K-Means 校准)
        max_sim, _ = sim_cos.max(dim=-1, keepdim=True)   # [B, N, 1]
        reliability = torch.sigmoid(
            self.rel_slope * (max_sim - self.rel_threshold)
        )  # [B, N, 1]

        # ===== 软路由 → 质心上下文 =====
        routing_temp = torch.exp(self.routing_log_temp)   # scalar
        soft_assign = F.softmax(sim_cos * routing_temp, dim=-1)  # [B, N, K]
        centroid_ctx = torch.matmul(soft_assign, self.centroids)  # [B, N, d_proj]

        # ===== 质心条件修正 MLP =====
        # 拼接归一化特征 + 质心上下文 → 修正向量
        corr_input = torch.cat([x_norm, centroid_ctx], dim=-1)  # [B, N, d_model+d_proj]
        correction = self.correction_out(
            self.correction_drop(self.correction_act(self.correction_in(corr_input)))
        )  # [B, N, d_model], 在归一化空间
        # 映射回原始尺度
        correction = correction * rev_std  # [B, N, d_model]

        # ===== 门控残差 =====
        gate = torch.sigmoid(self.gate_alpha)  # scalar, 初始 ≈ 0.018
        out = x + gate * correction

        # ===== Evidence Engine (诊断, 不进入主梯度路径) =====
        with torch.no_grad():
            Q_ctrl = Q.detach()
            Q_n_ctrl = F.normalize(Q_ctrl, p=2, dim=-1)
            sim_ctrl = torch.matmul(Q_n_ctrl, K_n.detach().T)
            adaptive_thr = sim_ctrl.mean(dim=-1, keepdim=True)
            dir_gate = self.softplus((sim_ctrl - adaptive_thr) * self.alpha_dir)
            s_ev = F.softplus(self.scale_evidence)
            evidence = dir_gate * s_ev
            S_train = torch.sum(evidence + 1.0, dim=-1, keepdim=True)
            u_std = self.n_centroids / S_train  # [B, N, 1]

        # ===== 辅助损失 =====
        # 1. 质心正交: 鼓励 K 个质心互相远离
        ortho_loss = F.mse_loss(
            torch.matmul(K_n, K_n.T),
            torch.eye(self.n_centroids, device=x.device)
        )
        # 2. 负载均衡: 鼓励所有质心被均匀使用 (防止质心退化)
        avg_routing = soft_assign.mean(dim=(0, 1))  # [K]
        load_balance = F.mse_loss(
            avg_routing,
            torch.ones_like(avg_routing) / self.n_centroids
        )
        geo_loss = torch.tensor(0.0, device=x.device)
        latent_loss = load_balance

        # ===== Debug 日志 =====
        self._fwd_count += 1
        if self.training and self._fwd_count % 200 == 1:
            with torch.no_grad():
                ms_flat = max_sim.squeeze(-1)
                rel_flat = reliability.squeeze(-1)
                corr_norm = (gate * correction).norm(dim=-1).mean().item()
                gate_val = gate.item()

                # 质心使用分布
                best_k = sim_cos.argmax(dim=-1)
                usage = torch.bincount(best_k.reshape(-1),
                                       minlength=self.n_centroids).float()
                usage_pct = usage / usage.sum() * 100
                usage_str = ','.join([f'{u:.0f}%' for u in usage_pct.tolist()])

                # 路由熵 (越高越均匀, 理想 = ln(K))
                routing_ent = -(avg_routing * torch.log(avg_routing + 1e-8)).sum().item()
                max_ent = math.log(self.n_centroids)

                # 质心正交度
                gram = torch.matmul(K_n, K_n.T)
                gram_off = gram - torch.eye(self.n_centroids, device=x.device)
                c_max_cos = gram_off.max().item()

                print(f"[GeoHCAN-MoE K={self.n_centroids}] "
                      f"gate:{gate_val:.4f} | "
                      f"||corr||:{corr_norm:.4f} | "
                      f"maxSim:{ms_flat.mean():.3f}"
                      f"[{ms_flat.min():.3f},{ms_flat.max():.3f}]")
                print(f"  routing: ent={routing_ent:.3f}/{max_ent:.3f} | "
                      f"usage=[{usage_str}] | "
                      f"L_bal={load_balance.item():.5f} | "
                      f"C_maxCos={c_max_cos:.3f}")
                print(f"  params: gate_α={self.gate_alpha.item():.3f} "
                      f"rout_T={routing_temp.item():.1f} "
                      f"thr={self.rel_threshold.item():.4f} "
                      f"slope={self.rel_slope.item():.1f} | "
                      f"L_ortho={ortho_loss.item():.4f}")

        return out, u_std, torch.tensor(0.0, device=x.device), \
            None, ortho_loss, geo_loss, latent_loss


# =====================================================================
#  Backbone — 几何注意力偏置 + MoE 修正 (双管齐下)
# =====================================================================
class Backbone(nn.Module):
    """
    iTransformer Backbone + GeoHCAN 双路增强:

    增强路径 1: 几何注意力偏置 (Geometric Attention Bias)
        在 Embedding 后、进入 Encoder 前, 计算变量间低维相似度矩阵,
        作为 attention bias 注入每一层的 softmax 之前.
        → 引导 encoder 关注几何上相似的变量 → 改变注意力模式
        → 使用 iTransformer 专属的 Geo{FullAttention,AttentionLayer,EncoderLayer,Encoder}
           子类, 不影响 layers/ 下的共享代码

    增强路径 2: 质心条件修正 MLP (Post-Encoder Correction)
        encoder 输出后, VariateGeoHCAN 计算质心条件修正.
        → 为不同质心模式的变量学习不同修正策略

    安全保证:
        - 注意力偏置: geo_bias_scale 初始 = 0 → bias = 0 → 与 baseline 完全一致
        - 修正 MLP: correction_out 零初始化 → correction = 0
        - 两条增强路径都从零起步, 通过梯度逐步学到有用信号
    """

    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ===== Embedding (与 standalone Model 完全相同) =====
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        # ===== Encoder: 使用 Geo 子类, 支持 attn_bias 透传 =====
        self.encoder = GeoEncoder(
            [
                GeoEncoderLayer(
                    GeoAttentionLayer(
                        GeoFullAttention(False, configs.factor,
                                         attention_dropout=configs.dropout,
                                         output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # ===== Projection: d_model → pred_len (与 standalone Model 完全相同) =====
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # ===== 增强路径 1: 几何注意力偏置 =====
        # 低维投影 → 变量间余弦相似度 → attention bias
        geo_bias_dim = 16
        self.geo_bias_proj = nn.Linear(configs.d_model, geo_bias_dim)
        # 偏置强度: 初始 = 0 → 无偏置 → 完全等价 baseline
        self.geo_bias_scale = nn.Parameter(torch.tensor(0.0))

        # ===== 增强路径 2: GeoHCAN MoE 修正 (post-encoder) =====
        self.geo_hcan = VariateGeoHCAN(
            d_model=configs.d_model,
            d_proj=getattr(configs, 'd_proj', 32),
            n_centroids=getattr(configs, 'num_fine', 6),
            gate_power=int(getattr(configs, 'gate_power', 2)),
            detach_temp=bool(getattr(configs, 'detach_temp', 1)),
            use_proj_ln=bool(getattr(configs, 'use_proj_ln', 1)),
        )

        # ===== Debug 计步器 =====
        self._bb_fwd_count = 0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y_true=None):
        # ===== RevIN Normalization =====
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # ===== Inverted Embedding =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N_tok, d_model]

        # ===== 增强路径 1: 几何注意力偏置 =====
        # 从 embedding 计算变量间低维相似度, 注入 encoder attention
        emb_proj = F.normalize(self.geo_bias_proj(enc_out), dim=-1)  # [B, N, 16]
        geo_sim = torch.matmul(emb_proj, emb_proj.transpose(-1, -2))  # [B, N, N]
        attn_bias = self.geo_bias_scale * geo_sim.unsqueeze(1)  # [B, 1, N, N]

        # ===== Encoder (带几何注意力偏置) =====
        enc_out, attns = self.encoder(enc_out, attn_mask=None, attn_bias=attn_bias)

        # ===== 增强路径 2: GeoHCAN MoE 修正 (NO detach) =====
        enc_enhanced, u_std, T_mean, _, ortho_loss, geo_loss, latent_loss = \
            self.geo_hcan(enc_out, y_true=y_true)

        # ===== Projection + Denorm =====
        dec_out = self.projection(enc_enhanced).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # ===== Debug: 注意力偏置状态 =====
        if self.training:
            self._bb_fwd_count += 1
            if self._bb_fwd_count % 200 == 1:
                with torch.no_grad():
                    bs = self.geo_bias_scale.item()
                    sim_mean = geo_sim.mean().item()
                    N_tok = geo_sim.shape[-1]
                    sim_off = geo_sim - torch.eye(N_tok, device=geo_sim.device)
                    sim_off_mean = sim_off.mean().item()
                    print(f"  [AttnBias] scale={bs:.4f} | "
                          f"sim_mean={sim_mean:.3f} | "
                          f"off_diag_mean={sim_off_mean:.3f}")
            return dec_out, ortho_loss, geo_loss, latent_loss
        return dec_out


# =====================================================================
#  Model — 纯净版 (standalone, --no_geohcan 使用)
# =====================================================================
class Model(nn.Module):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
