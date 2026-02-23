import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Backbone(nn.Module):
    """
    Informer Backbone (ProbSparse Attention Encoder-Decoder).
    输出 [B, pred_len, d_model] 特征，不含最终投影。

    接口约定:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, d_model]
    """

    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # ========== Encoder ==========
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=getattr(configs, 'output_attention', False)),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            [ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)]
            if getattr(configs, 'distil', True) else None,
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # ========== Decoder (projection=None, 由外部 GeoHCAN + Head 负责) ==========
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor,
                                      attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=None  # 不含投影层
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Args:
            x_enc:      [B, seq_len, enc_in]
            x_mark_enc: [B, seq_len, mark_dim]
            x_dec:      [B, label_len + pred_len, dec_in]
            x_mark_dec: [B, label_len + pred_len, mark_dim]

        Returns:
            [B, pred_len, d_model]
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # 截取预测部分
        return dec_out[:, -self.pred_len:, :]
