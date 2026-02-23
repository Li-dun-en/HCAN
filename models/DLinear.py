"""
DLinear: Decomposition Linear Model
Paper: https://arxiv.org/pdf/2205.13504.pdf
Source: https://github.com/thuml/Time-Series-Library

Architecture:
    Input → series_decomp (moving_avg) → seasonal + trend
    seasonal → Linear(seq_len → pred_len)
    trend    → Linear(seq_len → pred_len)
    output   = seasonal + trend  [B, pred_len, C]
"""

import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    DLinear standalone model (original implementation).
    Channel-independent: Linear operates on time dimension per channel.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, 'individual', False)
        self.channels = configs.enc_in

        self.decomposition = series_decomp(configs.moving_avg)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, seq_len]
        trend_init = trend_init.permute(0, 2, 1)        # [B, C, seq_len]

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [B, pred_len, C]

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, C]


class Backbone(nn.Module):
    """
    DLinear Backbone for GeoHCAN residual correction (TimesNet-style).

    DLinear outputs only C=7 dim predictions — too low-dimensional for
    GeoHCAN's geometric routing. Instead, use FusedAdapter dual-stream:

        Raw  branch: x_norm [B,T,C] → Linear(C→D)  (spatial geometry)
        Feat branch: prediction [B,P,C] → Linear(C→D)  (temporal context)
        Fusion: Add + LayerNorm → [B,P,D] → GeoHCAN → delta

    Wrapper (reuses TimesNet residual correction path):
        y_base = base_proj(prediction)              # DLinear's own prediction
        geo_in = FusedAdapter(x_norm, prediction)   # dual-stream → D dim
        delta  = Head(GeoHCAN(geo_in)) * res_gate   # GeoHCAN correction
        output = (y_base + delta) → denorm
    """

    returns_rev_stats = True

    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, 'individual', False)
        self.channels = configs.enc_in
        self.output_dim = configs.c_out

        self.decomposition = series_decomp(configs.moving_avg)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Returns:
            (prediction [B, P, C], (means, stdev))
            Same format as TimesNet Backbone — triggers residual correction path.
        """
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm = x_enc_norm / stdev

        seasonal_init, trend_init = self.decomposition(x_enc_norm)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        prediction = (seasonal_output + trend_output).permute(0, 2, 1)

        return prediction, (means, stdev)
