"""
ETSformer: Exponential Smoothing Transformers for Time-series Forecasting (ICLR 2023)
Paper: https://arxiv.org/abs/2202.01381

核心思想:
    将经典时间序列分解 (Level / Growth / Season) 与 Transformer 结合:
    1. GrowthLayer: 用指数平滑 Attention 提取阻尼趋势 (damped trend)
    2. FourierLayer: 用 Top-K 频率选择提取季节性模式
    3. LevelLayer: 用指数平滑提取水平分量
    4. Decoder: Growth 用 DampingLayer 外推, Season 直接从 Fourier 外推
    5. 最终预测 = Level(最后时刻) + Growth(外推) + Season(外推)

注意: ETSformer 要求 e_layers == d_layers

直接从 thuml/Time-Series-Library 复制, 仅保留 long_term_forecast 任务。
"""

import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.ETSformer_EncDec import (
    EncoderLayer, Encoder, DecoderLayer, Decoder, Transform
)


class Model(nn.Module):
    """
    ETSformer 原版完整实现 (standalone, 仅 long_term_forecast)。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        assert configs.e_layers == configs.d_layers, \
            "ETSformer requires e_layers == d_layers"

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in,
                    configs.seq_len, self.pred_len, configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out,
                    self.pred_len,
                    dropout=configs.dropout,
                ) for _ in range(configs.d_layers)
            ],
        )

        self.transform = Transform(sigma=0.2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds[:, -self.pred_len:, :]


class Backbone(nn.Module):
    """
    ETSformer Backbone for GeoHCAN integration.

    采用 Autoformer/FEDformer 相同的分解模式:
        growth + season → d_model 维特征 (类似 Autoformer 的 seasonal)
        level           → c_out 维基础水平 (类似 Autoformer 的 trend)

    数据流:
        Encoder → (level, growths, seasons)
        Decoder layers → growth_horizon + season_horizon [B, P, d_model]
        → (features [B, P, d_model], trend [B, 1, c_out])
        Wrapper: Adapter(features) → GeoHCAN → Head → output + trend

    output_dim = d_model, returns_rev_stats = False
    """

    returns_rev_stats = False

    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_dim = configs.d_model

        assert configs.e_layers == configs.d_layers, \
            "ETSformer requires e_layers == d_layers"

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in,
                    configs.seq_len, self.pred_len, configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out,
                    self.pred_len,
                    dropout=configs.dropout,
                ) for _ in range(configs.d_layers)
            ],
        )

        self.transform = Transform(sigma=0.2)

        print(f"[ETSformer Backbone] d_model={configs.d_model}, "
              f"output_dim={self.output_dim}, "
              f"mode=decomposition (features + level)")

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        Returns:
            (features [B, P, d_model], trend [B, 1, c_out])

            features: growth + season 在 d_model 空间的累加 (投影前),
                      送入 GeoHCAN 做几何增强
            trend:    level 分量 (最后时刻), 在 Head 输出后直接相加,
                      与 Autoformer 的 trend 处理方式完全一致
        """
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growth_repr = []
        season_repr = []
        for idx, layer in enumerate(self.decoder.layers):
            growth_horizon, season_horizon = layer(growths[idx], seasons[idx])
            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)

        features = sum(growth_repr) + sum(season_repr)  # [B, P, d_model]
        trend = level[:, -1:]                            # [B, 1, c_out]

        return features, trend
