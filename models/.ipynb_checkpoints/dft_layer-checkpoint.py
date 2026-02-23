import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicEvidentialLayer(nn.Module):
    """
    SP-DEFI+ Step 3 Fix: 
    1. Evidence Scaling: 除以 sqrt(d) 防止开局过分自信。
    2. Zero Initialization: 确保初始状态不破坏主干特征 (关键!)。
    """
    def __init__(self, d_model, num_classes_fine, num_classes_coarse, d_proj=64):
        super(DynamicEvidentialLayer, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes_fine 
        self.d_proj = d_proj
        
        # 1. 几何原型
        self.centroids = nn.Parameter(torch.randn(num_classes_fine, d_proj))
        self.key_proj = nn.Linear(d_proj, d_proj, bias=False) 
        self.query_proj = nn.Linear(d_model, d_proj) 
        
        # 2. 时序趋势
        self.linear_eta = nn.Linear(d_model, d_model)
        self.trend_gate_proj = nn.Linear(num_classes_fine, d_model)
        
        # 3. 偏移与语义
        self.offset_head = nn.Linear(d_model, num_classes_fine) 
        self.offset_mapper = nn.Linear(1, d_model) 
        self.semantic_map = nn.Linear(num_classes_fine, d_model, bias=False)
        
        # 4. 输出
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # ========== [关键修正] 初始化策略 ==========
        nn.init.xavier_uniform_(self.centroids)
        # 让 output_proj 初始为 0，保证训练初期 out = x (不捣乱)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        # offset 也先置零
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, x):
        B, L, D = x.size()
        
        # --- A. 相对特征提取 ---
        rev_mean = x.mean(dim=1, keepdim=True)
        rev_std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - rev_mean) / rev_std
        
        # --- B. 证据计算 (修正版) ---
        Q = self.query_proj(x_norm) 
        K = self.key_proj(self.centroids) 
        
        # 1. 信号强度
        signal_strength = torch.norm(Q, p=2, dim=-1, keepdim=True) 
        
        # 2. 形状匹配
        Q_n = F.normalize(Q, p=2, dim=-1)
        K_n = F.normalize(K, p=2, dim=-1)
        cosine_score = torch.matmul(Q_n, K_n.t()) 
        
        # 3. 生成证据 [FIX: Scaling]
        # 除以 sqrt(d_proj) (e.g. 8.0)，把证据值从 ~10 拉回到 ~1.0
        # 这样 u 就会在 0.5 左右，T 就在 0.55 左右，而不是 0.1
        scale_factor = math.sqrt(self.d_proj)
        evidence = F.relu(cosine_score) * (signal_strength / scale_factor)
        
        # 4. 不确定性
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        u = self.num_classes / S 
        
        # --- C. 动态治理 ---
        dynamic_temp = 0.1 + 0.9 * u 
        
        # Attention (Raw Score 也需要 scaling)
        raw_score = torch.matmul(Q, K.t()) / scale_factor
        attn_weights = F.softmax(raw_score / dynamic_temp, dim=-1) 
        
        # --- D. 特征重构 ---
        semantic_base = self.semantic_map(attn_weights)
        
        raw_offsets = torch.tanh(self.offset_head(x))
        weighted_offset = (raw_offsets * attn_weights).sum(dim=-1, keepdim=True)
        offset_feat = self.offset_mapper(weighted_offset)
        
        eta = self.linear_eta(x)
        trend_gate = torch.sigmoid(self.trend_gate_proj(attn_weights))
        gated_trend = eta * trend_gate
        
        # --- E. 最终注入 ---
        full_reconstruction = semantic_base + offset_feat + gated_trend
        
        main_gate = 1.0 - u
        
        # 因为 output_proj 初始为0，所以初期 out ≈ norm(x)
        # 随着训练进行，模型会慢慢学会注入有用的特征
        out = self.norm(x + self.output_proj(full_reconstruction) * main_gate)
        
        return out, attn_weights, raw_offsets, (rev_mean, rev_std)
