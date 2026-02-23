"""
Autoformer Backbone — 基于官方实现 (thuml/Time-Series-Library)

双模式:
    纯净基准线 (--no_geohcan):
        100% 官方架构，输出 [B, pred_len, c_out]

    GeoHCAN 模式:
        DecoderLayer 100% 官方 (趋势投影 d_model→c_out)
        去掉 Decoder 最终 seasonal 投影, seasonal 保留 d_model 维度
        返回 (seasonal [B,P,d_model], trend [B,P,c_out])
        由 Model Wrapper 组合: GeoHCAN(seasonal)→Head(512→7) + trend
"""

import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    my_Layernorm, series_decomp
)


class Backbone(nn.Module):
    """
    Autoformer Backbone.

    纯净模式:  output_dim = c_out,   返回 Tensor [B, P, c_out]
    GeoHCAN:   output_dim = d_model, 返回 (seasonal [B,P,d_model], trend [B,P,c_out])
    """

    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.pure_baseline = getattr(configs, 'no_geohcan', False)

        # ========== Decomposition ==========
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decomp = series_decomp(kernel_size)

        # ========== Embedding ==========
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # ========== Encoder (100% 官方) ==========
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor,
                                        attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # ========== Decoder (DecoderLayer 100% 官方) ==========
        decoder_layers = [
            DecoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(True, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                    configs.d_model, configs.n_heads
                ),
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                    configs.d_model, configs.n_heads
                ),
                configs.d_model,
                configs.c_out,       # 趋势投影始终 d_model→c_out (100% 官方)
                configs.d_ff,
                moving_avg=kernel_size,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.d_layers)
        ]

        if self.pure_baseline:
            # 100% 官方: Decoder 最终投影 seasonal d_model→c_out
            dec_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.output_dim = configs.c_out
        else:
            # GeoHCAN 模式: 不投影 seasonal, 保留 d_model 供 GeoHCAN 使用
            dec_projection = None
            self.output_dim = configs.d_model

        self.decoder = Decoder(
            decoder_layers,
            norm_layer=my_Layernorm(configs.d_model),
            projection=dec_projection
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ===== Decomposition Init (100% 官方) =====
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]],
            device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)

        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1
        )
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1
        )

        # ===== Encoder (100% 官方) =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # ===== Decoder (DecoderLayer 100% 官方) =====
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )

        if self.pure_baseline:
            # 100% 官方: seasonal [B,L,c_out] + trend [B,L,c_out]
            dec_out = trend_part + seasonal_part
            return dec_out[:, -self.pred_len:, :]
        else:
            # GeoHCAN 模式: 分别返回 seasonal 和 trend
            seasonal = seasonal_part[:, -self.pred_len:, :]  # [B, P, d_model]
            trend = trend_part[:, -self.pred_len:, :]        # [B, P, c_out]
            return seasonal, trend
