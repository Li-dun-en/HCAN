import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicEvidentialLayer(nn.Module):
    """
    SP-DEFI+ (Sorted-Projection Dynamic Evidential Fuzzy Inference with Entropy-Adaptive Reconstruction)
    
    核心特性：
    - Internal RevIN：建立相对坐标系，解决特征漂移
    - RBF Evidence：不归一化的径向基函数，实现"未知即为零"
    - Dynamic Temperature：基于不确定性动态调节分布锐化程度
    - Soft Reconstruction：Centroids + Offsets 实现精确特征重建
    - Entropy Gate：基于原始不确定性的物理门控
    """
    
    def __init__(self, d_model, num_classes_fine, num_classes_coarse, d_proj=64, rbf_gamma=1.0):
        super(DynamicEvidentialLayer, self).__init__()
        
        self.d_model = d_model
        self.num_classes_fine = num_classes_fine
        self.num_classes_coarse = num_classes_coarse
        self.d_proj = d_proj
        
        # ========== 核心参数 ==========
        # 1. 投影与原型
        self.proj_in = nn.Linear(d_model, d_proj)
        self.centroids_fine = nn.Parameter(torch.randn(num_classes_fine, d_proj))
        self.centroids_coarse = nn.Parameter(torch.randn(num_classes_coarse, d_proj))
        self.gamma = nn.Parameter(torch.tensor(rbf_gamma))
        
        # 2. 动态交互分支
        self.linear_eta = nn.Linear(d_model, d_model)
        self.semantic_map = nn.Linear(num_classes_fine, d_model, bias=False)
        
        # 3. [新增] 偏移预测与投影
        self.offset_head = nn.Linear(d_model, num_classes_fine)  # 预测 K 个标量偏移
        self.offset_proj = nn.Linear(1, d_model)  # 将加权标量偏移映射回 d_model
        
        # 4. [新增] 动态温度调节因子
        self.alpha_temp = nn.Parameter(torch.tensor(0.5))
        
        # 5. [新增] 锚点均值 buffer (由 init_defi_centroids_sorted 初始化)
        self.register_buffer('anchor_means', torch.zeros(num_classes_fine))
        
        # 6. 输出层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # ========== 初始化策略 ==========
        # [REVERT FIX] 恢复小权重初始化，防止开局梯度爆炸
        nn.init.xavier_uniform_(self.linear_eta.weight, gain=0.01)
        nn.init.xavier_uniform_(self.semantic_map.weight, gain=0.01)
        
        # offset_head 保持为 0
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def _get_evidence(self, x_proj, centroids):
        """
        RBF Evidence 计算（禁止 Softmax，实现"未知即为零"）
        
        Args:
            x_proj: [B, L, d_proj] 投影后的特征
            centroids: [K, d_proj] 聚类中心
            
        Returns:
            evidence: [B, L, K] RBF 证据值
        """
        # [B, L, 1, d_proj] - [1, 1, K, d_proj] -> [B, L, K]
        dist_sq = torch.sum(
            (x_proj.unsqueeze(-2) - centroids.view(1, 1, -1, self.d_proj)) ** 2, 
            dim=-1
        )
        gamma = F.softplus(self.gamma) + 1e-6
        # 纯 RBF，无 Softmax，实现"未知即为零"
        return torch.exp(-dist_sq / (2 * gamma ** 2))

    def forward(self, x):
        """
        SP-DEFI+ Forward 流程
        
        Args:
            x: [B, L, D] Decoder 输出特征
            
        Returns:
            out: [B, L, D] 增强后的特征
            belief_fine_raw: [B, L, K_fine] 原始 belief (未 sharpen)，用于 Loss
            belief_coarse: [B, L, K_coarse] 粗粒度 belief
            raw_offsets: [B, L, K_fine] tanh 后的偏移预测
            rev_stats: (rev_mean, rev_std) Internal RevIN 统计量，确保 Loss 归一化对齐
        """
        
        # ========== Step A: Internal RevIN + Projection ==========
        # 时间维度归一化，建立相对坐标系
        rev_mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        rev_std = x.std(dim=1, keepdim=True) + 1e-5  # [B, 1, D]
        x_norm = (x - rev_mean) / rev_std
        
        # 投影到低维空间
        x_proj = self.proj_in(x_norm)  # [B, L, d_proj]
        
        # ========== Step B: RBF Evidence (无 Softmax) + DST 推断 ==========
        ev_fine = self._get_evidence(x_proj, self.centroids_fine)  # [B, L, K]
        
        # DST (Dempster-Shafer Theory) 推断
        alpha_f = ev_fine + 1.0  # Dirichlet 参数
        S_f = torch.sum(alpha_f, dim=-1, keepdim=True)  # [B, L, 1]
        belief_fine_raw = ev_fine / S_f  # 原始 belief，用于 Loss
        uncertainty = self.num_classes_fine / S_f  # [B, L, 1] 不确定性
        
        # ========== Step C: Dynamic Temperature (熵自适应) ==========
        # 公式: T = clamp(1.0 - alpha * (1.0 - uncertainty), min=0.1)
        # 逻辑: 自信 (u→0) → T→0.1 (锐化); 不确定 (u→1) → T→1.0 (平滑)
        alpha_clamped = torch.clamp(self.alpha_temp, 0.0, 1.0)
        T = torch.clamp(1.0 - alpha_clamped * (1.0 - uncertainty), min=0.1)
        
        # 锐化后的权重 (用于重建，不用于 Loss)
        weights_sharp = F.softmax(torch.log(belief_fine_raw + 1e-8) / T, dim=-1)  # [B, L, K]
        
        # ========== Step D: Soft Reconstruction ==========
        # 1. 语义基底
        semantic_base = self.semantic_map(weights_sharp)  # [B, L, D]
        
        # 2. 偏移预测 (保留，仅为了计算 Loss 训练特征，不注入!)
        raw_offsets = torch.tanh(self.offset_head(x))  # [B, L, K]
        
        # [CRITICAL FIX] 强制禁用 Offset 注入
        # 旧代码: feat = semantic_base + self.offset_proj(...)
        # 新代码:
        feat = semantic_base  # 只注入纯净的语义原型
        
        # ========== Step E: Injection (恢复门控) ==========
        
        # [REVERT FIX] 恢复熵门控
        # 原因: 必须有 Gate 保护，否则初始噪声会摧毁 Backbone 的特征提取能力
        gate = (1.0 - uncertainty) ** 2  # [B, L, 1]
        
        # 特征被 Gate 调制
        gated_feat = feat * gate  # [B, L, D]
        
        # 趋势提取
        eta = self.linear_eta(x)  # [B, L, D]
        
        # 调制注入
        injection = eta * gated_feat  # [B, L, D]
        
        # 残差连接
        out = self.norm(x + injection)
        
        # ========== 辅助输出: Coarse Belief ==========
        ev_coarse = self._get_evidence(x_proj, self.centroids_coarse)  # [B, L, K_coarse]
        alpha_c = ev_coarse + 1.0
        S_c = torch.sum(alpha_c, dim=-1, keepdim=True)
        belief_coarse = ev_coarse / S_c
        
        return out, belief_fine_raw, belief_coarse, raw_offsets, (rev_mean, rev_std)


# =============================================================================
# [GeoHCAN] Geometry-Aware Evidential Network — Stage 4.5 (Dual-Track)
#
# Dual-Track Uncertainty:
#   Track A (Training): Standard DST  u_std = K/S         → for L_UAC, L_KL
#   Track B (Control):  Log-Rescaled  u_resp = 1/(1+ln(1+Σe))  → linear → T
#
# This decouples loss computation (needs exact DST) from temperature control
# (needs responsive, seed-stable uncertainty).
# =============================================================================
class GeoHCAN(nn.Module):
    """
    GeoHCAN Stage 4.5 — Dual-Track Uncertainty.

    Track A (Training): evidence → alpha → S → u_std = K/S
      Standard DST formula, feeds L_UAC and L_KL losses.
      u_std may be very small (~0.03) with dense evidence — that's OK,
      the losses use alpha and S directly, not u.

    Track B (Control): evidence → Σe → u_resp = 1/(1+ln(1+Σe))
      Log-rescaled uncertainty in [0.13, 0.26] — responsive and seed-stable.
      Drives Dynamic Temperature T via linear mapping (no sigmoid bifurcation).
    """

    def __init__(self, d_model: int, d_proj: int = 64, n_centroids: int = 4,
                 c_out: int = 7, lambda_geo: float = 0.1, lambda_latent: float = 0.01,
                 use_proj_ln: bool = True, detach_temp: bool = True,
                 use_energy_gate: bool = False, gate_power: int = 2):
        """
        Args:
            d_model:       Input / output feature dimension (e.g. 512)
            d_proj:        Geometric latent-space dimension (e.g. 64)
            n_centroids:   Number of geometric anchors (default 4 for fine-grained)
            c_out:         Output feature dimension (for L_geo target projection)
            lambda_geo:    Weight for L_geo (geometric classification loss)
            lambda_latent: Weight for L_latent (latent reconstruction loss)
            use_proj_ln:   是否在投影后加 LayerNorm (长序列锁 ||Q|| 量级，短序列关闭)
            detach_temp:   是否对温度控制轨 detach Q (长序列防方向操纵，短序列关闭)
            use_energy_gate: 是否用 ||Q|| 量级做能量门控补偿证据 (短序列开启，长序列关闭)
        """
        super(GeoHCAN, self).__init__()

        self.d_model = d_model
        self.d_proj = d_proj
        self.n_centroids = n_centroids
        self.lambda_geo = lambda_geo
        self.lambda_latent = lambda_latent
        self.use_proj_ln = use_proj_ln
        self.detach_temp = detach_temp
        self.use_energy_gate = use_energy_gate
        self.gate_power = gate_power

        # ========== Core Layers ==========
        # Projection: d_model → d_proj
        # use_proj_ln=True: + LayerNorm 锁定投影空间，防止 ||Q|| 无序膨胀 (长序列推荐)
        # use_proj_ln=False: 自由投影，保留原始特征强度 (短序列推荐)
        self.proj = nn.Linear(d_model, d_proj)
        self.proj_norm = nn.LayerNorm(d_proj) if use_proj_ln else nn.Identity()

        # Learnable Centroids [N, d_proj] — wide init for latent-space coverage
        self.centroids = nn.Parameter(torch.empty(n_centroids, d_proj))
        nn.init.uniform_(self.centroids, -3.0, 3.0)

        # Output Projection: d_proj → d_model
        self.out_proj = nn.Linear(d_proj, d_model)

        # ========== Offset MLP (The Muscle) ==========
        # Operates in d_proj space: Linear → Tanh → Linear(→ d_proj)
        # Then out_proj maps the sum (H_geo + offset) back to d_model
        self.offset_mlp = nn.Sequential(
            nn.Linear(d_proj, d_proj),
            nn.Tanh(),
            nn.Linear(d_proj, d_proj)
        )
        # CRITICAL: near-zero init so offset ≈ 0 at start
        nn.init.uniform_(self.offset_mlp[2].weight, -1e-5, 1e-5)
        nn.init.zeros_(self.offset_mlp[2].bias)

        # ========== Learnable Scales ==========
        # Selection Sharpness (Path A) — init 3.0 for meaningful T dynamic range
        # Effective sharpness = scale_logits / T:
        #   T=0.1 → 30 (fully peaked), T=0.6 → 5 (86% best), T=1.0 → 3 (70% best)
        # scale=10 killed T (always peaked); scale=1 too flat at T=1.0
        self.scale_logits = nn.Parameter(torch.tensor(3.0))
        # Evidence Gain (Path B) — kept at 5.0 for loss-function compatibility
        # Sparsity is achieved by the biased direction gate, not by reducing scale
        self.scale_evidence = nn.Parameter(torch.tensor(5.0))

        # ========== Learnable T Mapping (Auto-Calibrating Sigmoid) ==========
        # 替代硬编码线性映射，自适应 u_resp 范围变化（如 AGC 引入后范围偏移）
        # T = T_min + (1 - T_min) * σ(slope * (u_resp - threshold))
        self.T_threshold = nn.Parameter(torch.tensor(0.3))   # u_resp 中心点
        self.T_slope = nn.Parameter(torch.tensor(20.0))      # 响应灵敏度

        # ========== L_geo: Target Projection (仅在 lambda_geo > 0 时创建) ==========
        # 延迟创建避免改变核心参数的随机初始化序列
        self._c_out = c_out
        self.target_proj = None  # 按需在 enable_geo_loss() 中创建

        # Direction-gate sensitivity
        self.alpha = 5.0
        # Reusable Softplus (safe gradients, no saturation)
        self.softplus = nn.Softplus()

    # ------------------------------------------------------------------
    def enable_geo_loss(self):
        """按需创建 target_proj，不影响核心参数的随机初始化"""
        if self.target_proj is None:
            self.target_proj = nn.Linear(self._c_out, self.d_proj).to(self.centroids.device)
            nn.init.xavier_uniform_(self.target_proj.weight, gain=0.1)
            print(f"[GeoHCAN] target_proj created ({self._c_out} -> {self.d_proj})")

    # ------------------------------------------------------------------
    def _geo_loss(self, attn, y_true):
        """L_geo: KL(attention || soft_target_from_Y)"""
        if self.target_proj is None:
            return torch.tensor(0.0, device=attn.device)
        K_n = F.normalize(self.centroids.detach(), dim=-1)   # [N, d_proj] detached
        y_proj = self.target_proj(y_true)                     # [B, T, d_proj]
        y_n = F.normalize(y_proj, dim=-1)
        sim = torch.matmul(y_n, K_n.T)                       # [B, T, N]
        soft_target = F.softmax(sim * 3.0, dim=-1)            # tau=3.0 锐化
        return F.kl_div(
            torch.log(attn + 1e-8), soft_target,
            reduction='mean'  # mean over all B*T*N elements, 量级合理 (~0.1-0.5)
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y_true: torch.Tensor = None):
        """
        Args:
            x: [B, T, d_model]

        Returns:
            out:        [B, T, d_model]  Enhanced features (residual)
            u_std:      [B, T, 1]        Standard DST uncertainty (for L_UAC, L_KL)
            T_mean:     scalar           Mean dynamic temperature (driven by u_resp)
            attn:       [B, T, N]        Attention weights
            ortho_loss: scalar           Centroid-diversity regulariser
        """
        # ==================== Pre-computation ====================
        # RevIN: 时间维归一化，消除漂移，稳定几何空间
        rev_mean = x.mean(dim=1, keepdim=True)            # [B, 1, d_model]
        rev_std  = x.std(dim=1, keepdim=True) + 1e-5
        x_norm   = (x - rev_mean) / rev_std

        Q = self.proj_norm(self.proj(x_norm))              # [B, T, d_proj] LayerNorm 锁定量级
        Q_n = F.normalize(Q, p=2, dim=-1)                 # [B, T, d_proj]
        K_n = F.normalize(self.centroids, p=2, dim=-1)    # [N, d_proj]
        sim_cos = torch.matmul(Q_n, K_n.T)                # [B, T, N]  ∈ [-1, 1]

        # ==================== Path A: Selection (Pure Geometry) ====================
        s_logits = F.softplus(self.scale_logits)           # ensure > 0
        logits = sim_cos * s_logits                        # [B, T, N]

        # ==================== Path B: Evidence Engine ====================
        # detach_temp=True:  detach Q 切断主损失 → T 的梯度链 (长序列推荐)
        # detach_temp=False: 允许梯度流经温度控制 (短序列不需要额外保护)
        Q_ctrl = Q.detach() if self.detach_temp else Q
        Q_n_ctrl = F.normalize(Q_ctrl, p=2, dim=-1)             # [B, T, d_proj]
        sim_cos_ctrl = torch.matmul(Q_n_ctrl, K_n.T)            # [B, T, N]
        # 自适应阈值: 用每个 token 的质心均值余弦相似度作为基准
        # 物理意义: 只有比"平均水平"更对齐的质心才获得强证据
        # 自动适配不同 pred_len 的 maxSim 范围 (96:~0.5, 336:~0.3, 720:~0.2)
        adaptive_thr = sim_cos_ctrl.mean(dim=-1, keepdim=True)   # [B, T, 1]
        dir_gate = self.softplus((sim_cos_ctrl - adaptive_thr) * self.alpha)  # [B, T, N]
        s_ev = F.softplus(self.scale_evidence)                        # ensure > 0
        if self.use_energy_gate:
            # 能量门控: 用 ||Q|| 量级补偿证据强度 (短序列推荐)
            q_norm = torch.norm(Q_ctrl, dim=-1, keepdim=True)        # [B, T, 1]
            energy_gate = self.softplus(q_norm)                       # [B, T, 1]
            evidence = dir_gate * energy_gate * s_ev                  # [B, T, N]
        else:
            evidence = dir_gate * s_ev                                # [B, T, N]

        # ==================== Track A: 数学一致性轨道 (Training) ====================
        S_train = torch.sum(evidence + 1.0, dim=-1, keepdim=True)  # [B, T, 1]
        u_std = self.n_centroids / S_train                          # [B, T, 1]

        # ==================== Track B: 物理响应性轨道 (Control) ====================
        max_ev, _ = evidence.max(dim=-1, keepdim=True)              # [B, T, 1]
        u_resp = 1.0 / (1.0 + torch.log1p(max_ev * self.n_centroids))  # [B, T, 1]

        # Learnable Sigmoid T mapping: auto-calibrates to any u_resp range
        T = 0.1 + 0.9 * torch.sigmoid(self.T_slope * (u_resp - self.T_threshold))  # [B, T, 1]

        # ==================== Path C: Reconstruction ====================
        attn = F.softmax(logits / T, dim=-1)               # [B, T, N]
        H_geo = torch.matmul(attn, self.centroids)         # [B, T, d_proj]

        # Offset in d_proj space, then project sum to d_model
        # [FIX-B] 使用归一化 Q_n 替代原始 Q，切断 proj 层模长膨胀通道
        # offset 在单位球面上操作，||Q|| 的增长不再被主损失驱动
        offset = self.offset_mlp(Q_n)                      # [B, T, d_proj]
        H_recon = H_geo + offset                           # [B, T, d_proj] 几何重构
        H_final = self.out_proj(H_recon)                   # [B, T, d_model]

        # Uncertainty-gated residual: fall back to backbone when uncertain
        inject_gate = (1.0 - u_resp) ** self.gate_power      # [B, T, 1]  p=2: u=0.2→0.64; p=4: u=0.2→0.41
        out = inject_gate * H_final + x                    # [B, T, d_model]

        # ==================== Aux Loss: Orthogonal Centroids ====================
        gram = torch.matmul(K_n, K_n.T)                   # [N, N]
        ortho_loss = F.mse_loss(gram, torch.eye(self.n_centroids, device=x.device))

        # ==================== Aux Loss: L_latent (方向对齐，Stop-Gradient 防坍缩) ====================
        # 余弦相似度: 只要求重构方向与 Q 一致，不要求幅度匹配
        # Q.detach() 阻止梯度回流到 proj → 防止 Projection Collapse
        # 值域 [0, 2]，完美对齐=0，正交=1，反向=2
        latent_loss = 1.0 - F.cosine_similarity(H_recon, Q.detach(), dim=-1).mean()

        # ==================== Aux Loss: L_geo (几何分类，需外部启用) ====================
        geo_loss = torch.tensor(0.0, device=x.device)
        if y_true is not None and self.training:
            geo_loss = self._geo_loss(attn, y_true)

        # ==================== Debug (0.5 % sample) ====================
        if self.training and torch.rand(1).item() < 0.005:
            max_sim = sim_cos.max(dim=-1)[0].mean().item()
            print(f"[GeoHCAN N={self.n_centroids}] "
                  f"Ev:{evidence.mean().item():.1f} | "
                  f"maxSim:{max_sim:.3f} | "
                  f"u_resp:{u_resp.mean().item():.3f} | "
                  f"T:{T.mean().item():.3f} | "
                  f"gate:{inject_gate.mean().item():.3f} | "
                  f"||Q||:{torch.norm(Q, dim=-1).mean().item():.1f} | "
                  f"L_lat:{latent_loss.item():.4f} | "
                  f"L_geo:{geo_loss.item():.4f}")

        return out, u_std, T.mean(), attn, ortho_loss, geo_loss, latent_loss
