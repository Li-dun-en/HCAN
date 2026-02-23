"""
Non-stationary Transformer (NeurIPS 2022)
Paper: https://openreview.net/pdf?id=ucNDIDRNjjv

核心思想:
    时间序列的非平稳性 (均值/方差随时间变化) 会破坏 Attention 的分布假设。
    本文提出 De-stationary Attention:
        1. 对输入做 Instance Normalization (消除非平稳性)
        2. 用 MLP 从原始序列学习 tau (温度因子) 和 delta (偏移因子)
        3. 在 Attention 的 pre-softmax score 上: score = Q·K^T * tau + delta
           → tau 恢复被归一化抹去的统计量级差异
           → delta 恢复被归一化抹去的时间依赖偏移
        4. 输出时反归一化, 恢复原始量级

直接从 thuml/Time-Series-Library 复制, 仅保留 long_term_forecast 任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors (tau / delta).

    输入: series_conv(x_enc) concat std/mean → 2*enc_in 维
    输出: tau (scalar, 控制 Attention 温度) 或 delta (seq_len 维, 控制位置偏移)
    """

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len, out_channels=1, kernel_size=kernel_size,
            padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     [B, S, E]
        # stats: [B, 1, E]
        batch_size = x.shape[0]
        x = self.series_conv(x)            # [B, 1, E]
        x = torch.cat([x, stats], dim=1)   # [B, 2, E]
        x = x.view(batch_size, -1)         # [B, 2E]
        y = self.backbone(x)               # [B, output_dim]
        return y


class Model(nn.Module):
    """
    Non-stationary Transformer 原版完整实现 (standalone, 仅 long_term_forecast)。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        p_hidden_dims = getattr(configs, 'p_hidden_dims', [128, 128])
        p_hidden_layers = getattr(configs, 'p_hidden_layers', 2)

        self.tau_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers,
            output_dim=1)
        self.delta_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers,
            output_dim=configs.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        x_raw = x_enc.clone().detach()

        # --- Instance Normalization ---
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        # --- De-stationary factors ---
        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        # --- Encoder ---
        x_dec_new = torch.cat([
            x_enc[:, -self.label_len:, :],
            torch.zeros_like(x_dec[:, -self.pred_len:, :])
        ], dim=1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # --- Decoder ---
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                               tau=tau, delta=delta)

        # --- De-Normalization ---
        dec_out = dec_out * std_enc + mean_enc

        return dec_out[:, -self.pred_len:, :]


# =====================================================================
#  Backbone — 用于 GeoHCAN 外挂集成
#  输出 decoder 特征 [B, P, d_model] + RevIN 统计量
#  由 Wrapper 完成: adapter → GeoHCAN → Head → denorm
# =====================================================================
class Backbone(nn.Module):
    """
    Non-stationary Transformer Backbone for GeoHCAN integration.

    数据流:
        x_enc [B, L, C] → Normalize → Embedding → DSAttention Encoder
        → Decoder (无最终 projection) → features [B, label_len+pred_len, d_model]
        → 截取 [-P:] → [B, P, d_model]
        + RevIN stats (means, stdev)

    output_dim = d_model, returns_rev_stats = True
    Wrapper: adapter(d_model→geohcan_dim) → GeoHCAN → Head → denorm
    """

    returns_rev_stats = True

    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_dim = configs.d_model

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DSAttention(False, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=None  # 不做最终投影, 交给 Wrapper
        )

        p_hidden_dims = getattr(configs, 'p_hidden_dims', [128, 128])
        p_hidden_layers = getattr(configs, 'p_hidden_layers', 2)

        self.tau_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers,
            output_dim=1)
        self.delta_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers,
            output_dim=configs.seq_len)

        print(f"[NS-Transformer Backbone] d_model={configs.d_model}, "
              f"output_dim={self.output_dim}, returns_rev_stats=True")

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        Returns: (features [B, P, d_model], (means, stdev))
        """
        x_raw = x_enc.clone().detach()

        # ① Instance Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / stdev

        # ② De-stationary factors
        tau = self.tau_learner(x_raw, stdev)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, means)

        # ③ Encoder
        x_dec_new = torch.cat([
            x_enc[:, -self.label_len:, :],
            torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]],
                        device=x_enc.device)
        ], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # ④ Decoder (无 projection)
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                               tau=tau, delta=delta)

        # ⑤ 截取预测部分
        features = dec_out[:, -self.pred_len:, :]  # [B, P, d_model]

        return features, (means, stdev)
