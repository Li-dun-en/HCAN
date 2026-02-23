"""
FEDformer — Frequency Enhanced Decomposed Transformer
Paper: https://proceedings.mlr.press/v162/zhou22g.html
Source: https://github.com/thuml/Time-Series-Library

纯净 baseline 实现, 100% 忠于官方架构:
    Encoder: Fourier/Wavelet Enhanced Attention + Series Decomposition
    Decoder: Fourier/Wavelet Enhanced Cross-Attention + Series Decomposition + Trend Projection

两种版本:
    version='fourier'  — Fourier Enhanced Block (默认, 推荐)
    version='Wavelets' — Multi-Wavelet Transform

双模式:
    纯净基准线 (--no_geohcan):
        100% 官方架构, 输出 [B, pred_len, c_out]

    GeoHCAN 模式:
        去掉 Decoder 最终 seasonal 投影, seasonal 保留 d_model 维度
        返回 (seasonal [B,P,d_model], trend [B,P,c_out])
        由 Model Wrapper 组合: GeoHCAN(seasonal) → Head(d_model→c_out) + trend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    my_Layernorm, series_decomp
)


# ===================================================================
#  Model — 纯净 FEDformer (standalone, 用于 --no_geohcan 或独立运行)
# ===================================================================
class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain
    and achieves O(N) complexity.
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        Args:
            configs: argparse namespace with model hyperparameters
            version: 'fourier' or 'Wavelets'
            mode_select: 'random' or 'low' (frequency mode selection)
            modes: number of frequency modes to select
        """
        super(Model, self).__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # ========== Decomposition ==========
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decomp = series_decomp(kernel_size)

        # ========== Embedding ==========
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        # ========== Attention mechanism selection ==========
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=configs.d_model,
                base='legendre',
                activation='tanh')
        else:
            # Default: Fourier Enhanced Attention
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                n_heads=configs.n_heads,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                n_heads=configs.n_heads,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
                num_heads=configs.n_heads)

        # ========== Encoder ==========
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # ========== Decoder ==========
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # ========== Task-specific heads ==========
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Long/short term forecasting."""
        # Decomposition init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)

        # Decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Decoder
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # Final: trend + seasonal
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
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


# ===================================================================
#  Backbone — 用于 GeoHCAN 集成的 FEDformer 骨干
# ===================================================================
class Backbone(nn.Module):
    """
    FEDformer Backbone for GeoHCAN integration.

    纯净模式: output_dim = c_out,   返回 Tensor [B, P, c_out]
    GeoHCAN:  output_dim = d_model, 返回 (seasonal [B,P,d_model], trend [B,P,c_out])
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.pure_baseline = getattr(configs, 'no_geohcan', False)
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # ========== Decomposition ==========
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decomp = series_decomp(kernel_size)

        # ========== Embedding ==========
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        # ========== Attention mechanism selection ==========
        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=configs.d_model,
                base='legendre',
                activation='tanh')
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                n_heads=configs.n_heads,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                n_heads=configs.n_heads,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
                num_heads=configs.n_heads)

        # ========== Encoder ==========
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=kernel_size,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # ========== Decoder ==========
        decoder_layers = [
            DecoderLayer(
                AutoCorrelationLayer(
                    decoder_self_att,
                    configs.d_model, configs.n_heads),
                AutoCorrelationLayer(
                    decoder_cross_att,
                    configs.d_model, configs.n_heads),
                configs.d_model,
                configs.c_out,
                configs.d_ff,
                moving_avg=kernel_size,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.d_layers)
        ]

        if self.pure_baseline:
            dec_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.output_dim = configs.c_out
        else:
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
        seasonal_init, trend_init = self.decomp(x_enc)

        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # ===== Encoder =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # ===== Decoder =====
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        if self.pure_baseline:
            # 100% 官方: seasonal + trend → [B, L, c_out]
            dec_out = trend_part + seasonal_part
            return dec_out[:, -self.pred_len:, :]
        else:
            # GeoHCAN 模式: 分别返回 seasonal 和 trend
            seasonal = seasonal_part[:, -self.pred_len:, :]  # [B, P, d_model]
            trend = trend_part[:, -self.pred_len:, :]        # [B, P, c_out]
            return seasonal, trend
