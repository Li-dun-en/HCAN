import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from models.dft_layer import DynamicEvidentialLayer

class Model(nn.Module):
    """
    [Informer SP-DEFI 显式投影版]
    确保 Decoder 输出后经过明确的 Head 层，防止维度混乱。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # 参数绑定
        self.num_fine = getattr(configs, 'num_fine', 4)
        self.num_coarse = getattr(configs, 'num_coarse', 2)
        self.d_model = configs.d_model
        
        # 1. Encoder
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 2. Decoder
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=None # [关键] 这里设为 None，我们在 forward 里手动投影
        )

        # 3. SP-DEFI
        self.defi_layer = DynamicEvidentialLayer(d_model=self.d_model, num_classes_fine=self.num_fine, num_classes_coarse=self.num_coarse, d_proj=64, rbf_gamma=1.0)
        
        # 4. Head
        self.head = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # 1. Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # 2. Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        # 3. 截取预测部分 [B, pred_len, d_model]
        dec_feat = dec_out[:, -self.pred_len:, :]

        # 4. SP-DEFI+ 增强 (在 decoder 预测区间上)
        # 新接口: 移除 pred_len 参数，返回 5 个值
        enhanced_feat, b_fine, b_coarse, offsets, rev_stats = self.defi_layer(dec_feat)
        
        # 5. Head 投影 -> [B, pred_len, c_out]
        output = self.head(enhanced_feat)

        if self.training:
            # 返回 5 个值供 Loss 计算使用
            return output, b_fine, b_coarse, offsets, rev_stats
        else:
            return output