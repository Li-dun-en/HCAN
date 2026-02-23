"""
PatchTST Backbone — 基于官方实现 (yuqinie/patchtst)

双模式:
    纯净基准线 (--no_geohcan):
        100% 原版 PatchTST: Channel-Independent + FlattenHead + De-Normalization
        输出 [B, pred_len, c_out]

    GeoHCAN 模式 (Channel-Independent Inline):
        Encoder → GeoHCAN (直接在 B*C 维度操作) → FlattenHead → De-Norm
        GeoHCAN 以 B*C 为 batch, 对每个通道独立做时间模式增强
        零额外参数, 零信息损失, 与 PatchTST 的 CI 设计完全一致
        输出 [B, pred_len, c_out]  (对 Wrapper 透明)
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from models.dft_layer import GeoHCAN


class Transpose(nn.Module):
    """维度转置辅助模块 (用于 BatchNorm1d 维度适配)"""

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """PatchTST 展平预测头 — 每个通道独立预测"""

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Backbone(nn.Module):
    """
    PatchTST Backbone (Encoder-Only, Channel-Independent).

    双模式设计:

        纯净基准线 (--no_geohcan):
            RevIN → Patch → Encoder → FlattenHead → De-Norm

        GeoHCAN 模式 (Channel-Independent Inline):
            RevIN → Patch → Encoder → GeoHCAN → FlattenHead → De-Norm
            GeoHCAN 直接操作 [B*C, N, D], 以 B*C 为 batch
            逐通道时间模式增强, 零信息损失
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in

        self.pure_baseline = getattr(configs, 'no_geohcan', False)

        # 对 Wrapper: 两种模式都输出 c_out 维最终预测
        self.output_dim = configs.c_out

        # ---------- Patch 参数 ----------
        self.patch_len = getattr(configs, 'patch_len', patch_len)
        self.stride = getattr(configs, 'stride', stride)
        padding = self.stride

        # ---------- Patch Embedding ----------
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, padding, configs.dropout
        )

        # ---------- Transformer Encoder ----------
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2)
            )
        )

        # ---------- patch_num & FlattenHead ----------
        self.patch_num = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len,
            head_dropout=configs.dropout
        )

        # ---------- Inline GeoHCAN (仅 GeoHCAN 模式) ----------
        if not self.pure_baseline:
            self.geo_hcan = GeoHCAN(
                d_model=configs.d_model,
                d_proj=getattr(configs, 'd_proj', 64),
                n_centroids=getattr(configs, 'num_fine', 4),
                c_out=configs.c_out,
                lambda_geo=getattr(configs, 'lambda_geo', 0.1),
                lambda_latent=getattr(configs, 'lambda_latent', 0.01),
                use_proj_ln=bool(getattr(configs, 'use_proj_ln', 1)),
                detach_temp=bool(getattr(configs, 'detach_temp', 1)),
                use_energy_gate=bool(getattr(configs, 'use_energy_gate', 0)),
                gate_power=int(getattr(configs, 'gate_power', 2)),
            )
            print(f"[PatchTST] Inline GeoHCAN enabled: "
                  f"operates on [B×{configs.enc_in}, {self.patch_num}, {configs.d_model}]")

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                y_true=None):
        B, L, C = x_enc.shape

        # ===== ① RevIN 归一化 =====
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        # ===== ② Channel-Independent Patching =====
        x_enc = x_enc.permute(0, 2, 1)                           # [B, C, L]
        enc_out, n_vars = self.patch_embedding(x_enc)             # [B*C, N, D]

        # ===== ③ Transformer Encoder =====
        enc_out, attns = self.encoder(enc_out)                    # [B*C, N, D]

        # ===== ④ Inline GeoHCAN (仅 GeoHCAN 模式) =====
        # 直接在 [B*C, N, D] 上操作, B*C 当作 batch
        # 逐通道时间模式增强, 与 CI 设计完全一致
        ortho_loss = geo_loss = latent_loss = None
        if not self.pure_baseline:
            enc_out, u, T_mean, attn_w, ortho_loss, geo_loss, latent_loss = \
                self.geo_hcan(enc_out, y_true=y_true)             # [B*C, N, D]

        # ===== ⑤ Reshape + FlattenHead =====
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )                                                         # [B, C, N, D]
        enc_out = enc_out.permute(0, 1, 3, 2)                    # [B, C, D, N]

        dec_out = self.head(enc_out)                              # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                        # [B, pred_len, C]

        # ===== ⑥ De-Normalization =====
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 训练时返回 aux losses
        if ortho_loss is not None and self.training:
            return dec_out, ortho_loss, geo_loss, latent_loss

        return dec_out  # [B, pred_len, c_out]
