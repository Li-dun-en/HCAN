"""
models 包入口 — 统一模型构建接口

设计理念:
    GeoHCAN 是一个与主干无关的插件模块。
    任何主干只需实现 Backbone 类并注册到 backbone_dict，即可自动获得 GeoHCAN 增强。

    三种 GeoHCAN 集成模式:
        外挂模式 (Informer, Autoformer):
            Backbone → Wrapper 中的 GeoHCAN → Head → output
        残差修正模式 (TimesNet):
            Backbone → features [B,P,d] + rev_stats
            x_norm = RevIN(x_enc)
            geo_in = FusedAdapter(raw=x_norm, feat=features)  # Raw(C→D) + Feat(d→D)
            y_base = base_projection(features)                # TimesNet 自身预测
            delta  = Head(GeoHCAN(geo_in))                    # GeoHCAN 修正量
            output = (y_base + delta) → denorm                # 残差修正
        内嵌模式 (PatchTST Mid-Fusion):
            GeoHCAN 内置于 Backbone，Backbone 直接返回 (output, ortho, geo, latent)
            Wrapper 只做透传

    Backbone 接口约定:
        __init__(self, configs)
        forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec)
            → Tensor [B, P, c_out]                                    (纯净基准线)
            → (output, ortho_loss, geo_loss, latent_loss)             (内嵌 GeoHCAN)
            → (features [B,P,d_model], trend [B,P,c_out])            (Autoformer 分解)
            → (features [B,P,dim], (means, stdev))                    (RevIN 透传)
            → Tensor [B, P, d_model]                                  (外挂 GeoHCAN)

使用方式:
    from models import Model
    model = Model(args)
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dft_layer import GeoHCAN


# =====================================================================
#  融合适配器: Raw + Temporal 双流 → GeoHCAN (TimesNet 专用)
# =====================================================================
class FusedAdapter(nn.Module):
    """
    融合适配器: 结合原始通道特征和 backbone 时序特征，为 GeoHCAN 提供互补输入。

    Raw Branch  — Linear(C → D): 提供空间/通道几何信息 (绝对数值、相对比例)
    Feat Branch — Linear(d → D): 提供时序上下文 (周期相位、波峰波谷)
    Fusion      — Add + LayerNorm: 互补融合

    核心原理:
        TimesNet (16维) 告诉 GeoHCAN: "现在是波峰还是波谷" (时序上下文)
        Raw Input (7维)  告诉 GeoHCAN: "现在的绝对数值和相对比例" (空间几何坐标)
        GeoHCAN: 结合两者，发现 "在波峰(Time) + A比B高太多(Space) → 需要修正(Delta)"

    有效秩: rank(Raw) ≤ C=7, rank(Feat) ≤ d=16 → 融合后最高 23 维有效信息
    当 D=32 时，利用率 72% (远优于之前 16/512=3%)

    只做 Channel 维度投影，不做 Time 维度投影:
      对 Time 维做 Linear 会破坏局部性、引入未来信息泄漏风险;
      GeoHCAN 自带 Attention / DFT，不需要 Time 维预处理。
    """
    def __init__(self, enc_in, backbone_dim, out_dim, dropout=0.1):
        super().__init__()
        self.raw_proj  = nn.Linear(enc_in, out_dim)          # Raw:  C → D
        self.feat_proj = nn.Linear(backbone_dim, out_dim)    # Feat: d → D
        self.fusion_ln = nn.LayerNorm(out_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x_raw_norm, backbone_feat):
        """
        Args:
            x_raw_norm:    [B, T, C]  RevIN 归一化后的原始输入
            backbone_feat: [B, P, d]  backbone 输出的时序特征
        Returns:
            geo_in: [B, P, D]  融合特征 (送入 GeoHCAN)
        """
        P = backbone_feat.shape[1]
        T = x_raw_norm.shape[1]

        # ① Raw Branch: 通道维投影 [B, T, C] → [B, T, D]
        raw_feat = self.raw_proj(x_raw_norm)                  # [B, T, D]

        # ② 时间维对齐 (不用 Linear, 保持局部性)
        if T >= P:
            raw_feat = raw_feat[:, -P:, :]                    # 截取最后 P 步
        else:
            # T < P (pred_len > seq_len): 线性插值代替重复填充
            # 将 T 个真实数据点平滑分布到 P 个位置
            # 保留通道间几何比例关系, 消除重复填充的 "死区"
            # 旧方案(重复末端)在 pred_len=336 时 71% 是相同向量,
            # 污染 K-Means 初始化且 GeoHCAN 无法学习几何结构
            raw_feat = F.interpolate(
                raw_feat.transpose(1, 2),                     # [B, D, T]
                size=P, mode='linear', align_corners=False
            ).transpose(1, 2)                                 # [B, P, D]

        # ③ Feat Branch: 时序特征投影 [B, P, d] → [B, P, D]
        temp_feat = self.feat_proj(backbone_feat)              # [B, P, D]

        # ④ Fusion: Add + LayerNorm
        geo_in = self.fusion_ln(raw_feat + temp_feat)
        geo_in = self.dropout(geo_in)

        return geo_in


# =====================================================================
#  时序卷积适配器: 低维 backbone 专用 (DLinear 等 C≤7)
# =====================================================================
class TemporalConvAdapter(nn.Module):
    """
    低维 backbone 的时序特征提取器。

    FusedAdapter 对 DLinear 失效原因:
        两条 Linear(7→D) 流产生 rank≤14 的特征, 在 D=128 空间中
        仅 11% 维度携带信息, cosine similarity 被噪声淹没 (maxSim~0.29)。

    本适配器用 1D 卷积捕获局部时序模式 + 跨通道交互:
        Conv1d(C→D, k=3) → GELU → Conv1d(D→D, k=3) → Linear(T→P) → LN
        每个位置的感受野 = 5 步 × C 通道, 有效秩 ≤ min(5C, D)
        当 C=7, D=64 时: 有效秩≤35, 利用率 55% (远优于 FusedAdapter 的 11%)

    接口兼容 FusedAdapter (backbone_feat 参数被忽略, 因为原始时序已包含全部信息)。
    """
    def __init__(self, enc_in, out_dim, seq_len, pred_len, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(enc_in, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(out_dim)
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_raw_norm, backbone_feat=None):
        """
        Args:
            x_raw_norm:    [B, T, C]  RevIN 归一化后的原始输入
            backbone_feat: ignored (接口兼容, DLinear 预测已蕴含于原始输入)
        Returns:
            geo_in: [B, P, D]  时序卷积特征
        """
        h = x_raw_norm.permute(0, 2, 1)       # [B, C, T]
        h = F.gelu(self.conv1(h))              # [B, D, T]
        h = self.conv2(h)                      # [B, D, T]
        h = self.time_proj(h)                  # [B, D, P]
        h = h.permute(0, 2, 1)                # [B, P, D]
        return self.dropout(self.norm(h))


# =====================================================================
#  非线性适配器: 膨胀-激活-投影 (Informer/Autoformer 可选)
# =====================================================================
class NonlinearAdapter(nn.Module):
    """非线性适配器 (Informer/Autoformer 维度不匹配时可选使用)"""
    def __init__(self, in_dim, out_dim, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
#  主干注册表 — (模块路径, 类名)
# =====================================================================
backbone_dict = {
    'Informer':      ('models.Informer',      'Backbone'),
    'PatchTST':      ('models.PatchTST',      'Backbone'),
    'Autoformer':    ('models.Autoformer',     'Backbone'),
    'TimesNet':      ('models.TimesNet',       'Backbone'),
    'iTransformer':  ('models.iTransformer',   'Backbone'),
    'FEDformer':     ('models.FEDformer',      'Backbone'),
    'DLinear':       ('models.DLinear',        'Backbone'),
    'Nonstationary_Transformer': ('models.Nonstationary_Transformer', 'Backbone'),
    'ETSformer':  ('models.ETSformer',  'Backbone'),
}


def _get_backbone_cls(name):
    """按名称获取 Backbone 类 (惰性导入)"""
    if name not in backbone_dict:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(backbone_dict.keys())}"
        )
    module_path, class_name = backbone_dict[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# =====================================================================
#  统一模型: Backbone + (可选) GeoHCAN + Head
# =====================================================================
class Model(nn.Module):
    """
    统一模型包装器

    数据流:

        TimesNet + GeoHCAN (残差修正):
          Backbone → (features [B,P,16], rev_stats)
          x_norm = RevIN(x_enc)                                      [B,T,7]
          geo_in = FusedAdapter(raw=x_norm, feat=features)           [B,P,D]
          y_base = base_proj(features)                      → [B,P,7]  TimesNet 预测
          delta  = Head(GeoHCAN(geo_in))                    → [B,P,7]  GeoHCAN 修正
          output = (y_base + delta) * stdev + means         → [B,P,7]  最终输出

        Autoformer / FEDformer + GeoHCAN (外挂, 分解模式):
          Backbone → (seasonal [B,P,d_model], trend [B,P,c_out])
          seasonal → Adapter(d→D) → GeoHCAN → Head(D→c_out) → + trend → output

        DLinear + GeoHCAN (残差修正, TemporalConvAdapter):
          Backbone → (prediction [B,P,C=7], (means, stdev))
          x_norm = RevIN(x_enc)
          geo_in = TemporalConvAdapter(x_norm)   [B,P,D]  (Conv1d 提取时序模式)
          y_base = base_proj(prediction)          DLinear 自身预测
          delta  = Head(GeoHCAN(geo_in)) * res_gate   GeoHCAN 修正
          output = (y_base + delta) → denorm
          注: DLinear backbone_out=7 太低, FusedAdapter 的 Linear(7→D) 产生
              rank-deficient 特征导致 GeoHCAN 路由失效; Conv1d 有效秩 ≤ 5C=35

        Informer + GeoHCAN (外挂):
          Backbone → features [B,P,512]
          features → GeoHCAN → Head(512→7) → output [B,P,7]

        PatchTST + GeoHCAN (内嵌 Mid-Fusion):
          Backbone 内部: Encoder → Bridge → GeoHCAN → InvBridge → FlattenHead → De-Norm
          Backbone → (output [B,P,7], ortho, geo, latent)
          Wrapper 直接透传

        纯净基准线 (--no_geohcan):
          Backbone → output (按骨干原始架构直出)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.use_geohcan = not getattr(configs, 'no_geohcan', False)

        # GeoHCAN 工作维度 (允许与 backbone d_model 不同, 如 TimesNet d_model=16)
        _geohcan_d = getattr(configs, 'geohcan_d_model', 0)
        self._geohcan_dim = _geohcan_d if _geohcan_d > 0 else configs.d_model

        # ========== 特殊路径: 独立模型 (原版 standalone, 不经 GeoHCAN) ==========
        _standalone_models = {
            'TimesNet':     ('models.TimesNet',     'Model'),
            'iTransformer': ('models.iTransformer', 'Model'),
            'FEDformer':    ('models.FEDformer',    'Model'),
            'DLinear':      ('models.DLinear',      'Model'),
            'Nonstationary_Transformer': ('models.Nonstationary_Transformer', 'Model'),
            'ETSformer':  ('models.ETSformer',  'Model'),
        }
        self._is_standalone = (configs.model in _standalone_models and not self.use_geohcan)
        if self._is_standalone:
            _mod_path, _cls_name = _standalone_models[configs.model]
            _mod = importlib.import_module(_mod_path)
            _StandaloneModel = getattr(_mod, _cls_name)
            self.standalone_model = _StandaloneModel(configs)
            self.backbone = None
            self.adapter = None
            self.head = None
            self._has_internal_geohcan = False
            self._needs_denorm = False
            print(f"[Model] {configs.model} | standalone (原版, d_model={configs.d_model}, "
                  f"d_ff={configs.d_ff})")
            return

        # ========== 1. Backbone ==========
        backbone_cls = _get_backbone_cls(configs.model)
        self.backbone = backbone_cls(configs)
        backbone_out_dim = getattr(self.backbone, 'output_dim', configs.d_model)

        # 检测特殊模式
        self._has_internal_geohcan = hasattr(self.backbone, 'geo_hcan')
        self._needs_denorm = getattr(self.backbone, 'returns_rev_stats', False)

        # ========== 2. 外挂 GeoHCAN 模式 (Informer, Autoformer, TimesNet) ==========
        if self.use_geohcan and not self._has_internal_geohcan:

            if self._needs_denorm:
                _low_dim_backbone = (backbone_out_dim <= configs.enc_in)
                if _low_dim_backbone:
                    # ---- DLinear 等低维 backbone: TemporalConvAdapter ----
                    # backbone_out ≤ enc_in 时, FusedAdapter 的 Linear(C→D) 产生
                    # rank-deficient 特征; 改用 1D 卷积提取有效时序模式
                    self.adapter = TemporalConvAdapter(
                        enc_in=configs.enc_in,
                        out_dim=self._geohcan_dim,
                        seq_len=configs.seq_len,
                        pred_len=configs.pred_len,
                        dropout=configs.dropout,
                    )
                    adapter_name = (f"TemporalConvAdapter(C={configs.enc_in}"
                                    f"→{self._geohcan_dim})")
                else:
                    # ---- TimesNet: FusedAdapter (Raw + Feat 双流融合) ----
                    self.adapter = FusedAdapter(
                        enc_in=configs.enc_in,
                        backbone_dim=backbone_out_dim,
                        out_dim=self._geohcan_dim,
                        dropout=configs.dropout,
                    )
                    adapter_name = (f"FusedAdapter(raw={configs.enc_in}"
                                    f",feat={backbone_out_dim}→{self._geohcan_dim})")
                self.base_projection = nn.Linear(
                    backbone_out_dim, configs.c_out, bias=True)
                mode_str = (f"ResidualPlugin: base_proj({backbone_out_dim}→"
                            f"{configs.c_out}) + {adapter_name} → "
                            f"GeoHCAN → delta → y_base+delta → denorm")
            else:
                # ---- Informer / Autoformer: 普通 adapter ----
                _adapter_hidden = getattr(configs, 'adapter_hidden', 0)
                if backbone_out_dim != self._geohcan_dim:
                    if _adapter_hidden > 0:
                        self.adapter = NonlinearAdapter(
                            backbone_out_dim, self._geohcan_dim,
                            hidden_dim=_adapter_hidden,
                            dropout=configs.dropout,
                        )
                    else:
                        self.adapter = nn.Linear(backbone_out_dim, self._geohcan_dim)
                else:
                    self.adapter = None
                if self.adapter is not None and _adapter_hidden > 0:
                    adapter_info = (f"NonlinearAdapter({backbone_out_dim}→"
                                    f"{_adapter_hidden}→{self._geohcan_dim})")
                elif self.adapter is not None:
                    adapter_info = f"LinearAdapter({backbone_out_dim}→{self._geohcan_dim})"
                else:
                    adapter_info = "no adapter"
                mode_str = (f"Backbone({backbone_out_dim}) → {adapter_info} "
                            f"→ GeoHCAN → Head → c_out({configs.c_out})")

            self.geo_hcan = GeoHCAN(
                d_model=self._geohcan_dim,
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
            self.head = nn.Linear(self._geohcan_dim, configs.c_out, bias=True)
            # ReZero 机制 (TimesNet 残差修正专用):
            #   delta_head 正常 Xavier 初始化 → GeoHCAN 内部有梯度流动
            #   res_gate 初始化为 0 → delta 初始为 0, output≈y_base
            #   训练时 res_gate 自动学习何时/多大程度让 GeoHCAN 介入
            if self._needs_denorm:
                nn.init.xavier_uniform_(self.head.weight)
                nn.init.zeros_(self.head.bias)
                self.res_gate = nn.Parameter(torch.zeros(1))
        elif self._has_internal_geohcan:
            # 内嵌模式: Backbone 自带 GeoHCAN, Wrapper 不需要额外模块
            self.adapter = None
            self.head = None
            mode_str = "Backbone(内嵌 GeoHCAN)"
        else:
            # ========== 3. 纯净基准线 ==========
            self.adapter = None
            if backbone_out_dim != configs.c_out:
                self.head = nn.Linear(backbone_out_dim, configs.c_out, bias=True)
                mode_str = "Backbone + Head (pure baseline)"
                if self._needs_denorm:
                    mode_str += " + RevIN denorm"
            else:
                self.head = None
                mode_str = "Backbone (pure official baseline)"

        print(f"[Model] {configs.model} | backbone_out={backbone_out_dim} "
              f"| geohcan_dim={self._geohcan_dim} | {mode_str}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                y_true=None):
        """
        统一前向接口

        Returns:
            训练 (GeoHCAN): (output, ortho_loss, geo_loss, latent_loss)
            其他:           output [B, pred_len, c_out]
        """

        # ========== TimesNet 独立模型 (直接透传) ==========
        if self._is_standalone:
            return self.standalone_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # ========== 内嵌 GeoHCAN 模式 (PatchTST Mid-Fusion) ==========
        if self._has_internal_geohcan:
            # Backbone 直接返回最终预测 (训练时含 aux losses)
            return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                 y_true=y_true)

        # ========== 外挂 GeoHCAN 或纯净基准线 ==========
        backbone_out = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 解析 backbone 输出
        trend = None
        rev_stats = None
        if isinstance(backbone_out, tuple) and self._needs_denorm:
            # TimesNet 等: (features, (means, stdev))
            dec_feat, rev_stats = backbone_out
        elif isinstance(backbone_out, tuple):
            dec_feat, trend = backbone_out  # Autoformer: (seasonal, trend)
        else:
            dec_feat = backbone_out

        if self.use_geohcan:

            if rev_stats is not None:
                # ============ TimesNet 残差修正模式 ============
                # FusedAdapter: Raw(x_norm) + Feat(backbone) → geo_in
                means, stdev = rev_stats
                x_norm = (x_enc - means) / stdev              # [B, T, C]
                geo_in = self.adapter(x_norm, dec_feat)        # [B, P, D]

                # GeoHCAN → [B, P, geohcan_dim]
                enhanced_feat, u, T_mean, attn, ortho_loss, geo_loss, latent_loss = \
                    self.geo_hcan(geo_in, y_true=y_true)

                # y_base: TimesNet 自身的预测 (normalized space)
                # delta:  GeoHCAN 产生的修正量 × ReZero 门控
                #   res_gate 初始=0 → delta=0 → output=y_base (baseline)
                #   训练中 res_gate 自动打开 → GeoHCAN 逐渐介入修正
                y_base    = self.base_projection(dec_feat)     # [B, P, c_out]
                delta_raw = self.head(enhanced_feat)           # [B, P, c_out]
                delta     = delta_raw * self.res_gate          # ReZero 门控
                output    = y_base + delta                     # 残差修正

                # RevIN De-Normalization
                output = output * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
                output = output + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            else:
                # ============ Informer / Autoformer: 原有架构 ============
                if self.adapter is not None:
                    geo_in = self.adapter(dec_feat)
                else:
                    geo_in = dec_feat

                enhanced_feat, u, T_mean, attn, ortho_loss, geo_loss, latent_loss = \
                    self.geo_hcan(geo_in, y_true=y_true)

                output = self.head(enhanced_feat)
                if trend is not None:
                    output = output + trend

            if self.training:
                return output, ortho_loss, geo_loss, latent_loss
            return output
        else:
            # 纯净基准线
            if self.head is not None:
                output = self.head(dec_feat)
            else:
                output = dec_feat

            # RevIN De-Normalization (如 TimesNet baseline)
            if rev_stats is not None:
                means, stdev = rev_stats
                output = output * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
                output = output + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

            return output
