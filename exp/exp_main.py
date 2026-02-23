from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Model   # 统一模型入口
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
plt.switch_backend('agg')  # 无 GUI 环境使用


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # =============================================
        # 统一模型构建: Backbone + GeoHCAN + Head
        # Backbone 由 args.model 选择 (Informer / PatchTST / Autoformer / ...)
        # =============================================
        model_instance = Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model_instance = nn.DataParallel(model_instance, device_ids=self.args.device_ids)
        return model_instance

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # GeoHCAN 参数单独设更高 LR (从零初始化需要更强信号)
        if (not getattr(self.args, 'no_geohcan', False)
                and hasattr(model_ref, 'backbone')
                and hasattr(model_ref.backbone, 'geo_hcan')):
            geo_names = {'geo_hcan', 'geo_bias'}
            geo_params, base_params = [], []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any(gn in name for gn in geo_names):
                        geo_params.append(param)
                    else:
                        base_params.append(param)
            geo_lr = self.args.learning_rate * 10  # 10x LR
            model_optim = optim.Adam([
                {'params': base_params, 'lr': self.args.learning_rate},
                {'params': geo_params, 'lr': geo_lr},
            ])
            print(f">>> [Optimizer] base_lr={self.args.learning_rate}, "
                  f"geo_lr={geo_lr} (10x) | "
                  f"base_params={len(base_params)}, geo_params={len(geo_params)}")
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    # =========================================================================
    #  [GeoHCAN] K-Means 质心初始化 — 通过统一 Backbone 接口，与主干无关
    # =========================================================================
    def init_geo_centroids(self, train_loader):
        """
        K-Means 初始化 GeoHCAN centroids.

        VariateGeoHCAN 模式 (iTransformer):
            需要运行 Encoder 获取变量表征 [B, N_tok, d_model],
            然后 VariateGeoHCAN 内部 RevIN + proj 获取 d_proj 特征,
            K-Means 找到 K 个正常模式质心.

        外挂模式 (Informer/Autoformer/TimesNet):
            通过 backbone 前向获取特征, 再投影.
        """
        self.model.eval()
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        proj_features = []

        with torch.no_grad():
            for i, (bx, by, bxm, bym) in enumerate(train_loader):
                if i >= 50:
                    break
                bx = bx.float().to(self.device)
                bxm = bxm.float().to(self.device)
                bym = bym.float().to(self.device)

                # ===== 获取 GeoHCAN 输入特征 =====
                _has_internal = hasattr(model_ref.backbone, 'geo_hcan')
                if _has_internal:
                    bb = model_ref.backbone
                    _geo = bb.geo_hcan

                    # RevIN 归一化 (与 Backbone.forward 完全一致)
                    _x = bx
                    _means = _x.mean(dim=1, keepdim=True).detach()
                    _x = _x - _means
                    _stdev = torch.sqrt(
                        torch.var(_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                    _x = _x / _stdev

                    if hasattr(bb, 'patch_embedding'):
                        # PatchTST 内嵌: RevIN + Patching + Encoder
                        _x_p = _x.permute(0, 2, 1)
                        _enc, _nvars = bb.patch_embedding(_x_p)
                        _enc, _ = bb.encoder(_enc)
                        dec_feat = _enc  # [B*C, N, D]
                        # PatchTST 模式: 用 encoder 输出做 RevIN + proj
                        rev_mean = dec_feat.mean(dim=1, keepdim=True)
                        rev_std = dec_feat.std(dim=1, keepdim=True) + 1e-5
                        f_proj = _geo.proj_norm(
                            _geo.proj((dec_feat - rev_mean) / rev_std)
                        )
                    else:
                        # VariateGeoHCAN (iTransformer):
                        # 需要运行 Encoder 获取变量表征
                        _enc = bb.enc_embedding(_x, bxm)  # [B, N_tok, d_model]
                        _enc, _ = bb.encoder(_enc, attn_mask=None)
                        # VariateGeoHCAN 内部 RevIN + proj
                        _rev_mean = _enc.mean(dim=1, keepdim=True)
                        _rev_std = _enc.std(dim=1, keepdim=True) + 1e-5
                        _enc_norm = (_enc - _rev_mean) / _rev_std
                        f_proj = _geo.proj_norm(
                            _geo.proj(_enc_norm)           # [B, N_tok, d_proj]
                        )

                    proj_features.append(
                        f_proj.reshape(-1, f_proj.shape[-1]).cpu().numpy()
                    )
                else:
                    # 外挂模式: backbone 输出 + adapter + proj
                    by_dec = torch.zeros(
                        [bx.shape[0], self.args.pred_len, by.shape[2]]
                    ).float().to(self.device)
                    by_dec = torch.cat(
                        [by[:, :self.args.label_len, :].float().to(self.device),
                         by_dec], dim=1
                    )
                    backbone_out = model_ref.backbone(bx, bxm, by_dec, bym)
                    rev_stats = None
                    if isinstance(backbone_out, tuple) and model_ref._needs_denorm:
                        dec_feat, rev_stats = backbone_out
                    elif isinstance(backbone_out, tuple):
                        dec_feat = backbone_out[0]
                    else:
                        dec_feat = backbone_out

                    if model_ref.adapter is not None:
                        if rev_stats is not None:
                            means, stdev = rev_stats
                            x_norm = (bx - means) / stdev
                            dec_feat = model_ref.adapter(x_norm, dec_feat)
                        else:
                            dec_feat = model_ref.adapter(dec_feat)

                    geo_hcan_ref = model_ref.geo_hcan
                    rev_mean = dec_feat.mean(dim=1, keepdim=True)
                    rev_std = dec_feat.std(dim=1, keepdim=True) + 1e-5
                    f_proj = geo_hcan_ref.proj_norm(
                        geo_hcan_ref.proj((dec_feat - rev_mean) / rev_std)
                    )
                    proj_features.append(
                        f_proj.reshape(-1, f_proj.shape[-1]).cpu().numpy()
                    )

        all_proj = np.concatenate(proj_features, axis=0)
        _geo = model_ref.backbone.geo_hcan if hasattr(model_ref.backbone, 'geo_hcan') \
            else model_ref.geo_hcan
        n_centroids = _geo.n_centroids
        kmeans = KMeans(n_clusters=n_centroids, n_init=10, random_state=42).fit(all_proj)
        _geo.centroids.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(self.device)

        # 验证: 计算初始化后的 cosine similarity 分布
        with torch.no_grad():
            all_proj_t = torch.tensor(all_proj, dtype=torch.float32)
            Q_n = F.normalize(all_proj_t, p=2, dim=-1)
            K_n = F.normalize(_geo.centroids.cpu(), p=2, dim=-1)
            sim_all = torch.matmul(Q_n, K_n.T)
            max_sim_all = sim_all.max(dim=-1)[0]

            # ===== 校准 Reliability Gate 阈值 (仅 UncertaintyIsolation 模式) =====
            if hasattr(_geo, 'rel_threshold'):
                # 设 threshold = p1(max_sim) * 0.5, 极端保守
                # 确保 99%+ 正常数据 rel ≈ 1, 干净数据几乎不修改
                p1 = np.percentile(max_sim_all.numpy(), 1)
                new_thr = float(p1 * 0.5)
                _geo.rel_threshold.data.fill_(new_thr)
                # 设 slope 使得 p10(max_sim) 对应 reliability ≈ 0.999
                p10 = float(np.percentile(max_sim_all.numpy(), 10))
                med = float(np.median(max_sim_all.numpy()))
                margin = max(p10 - new_thr, 0.05)
                # sigmoid(slope * margin) = 0.999 → slope * margin ≈ 6.9
                new_slope = 6.9 / margin
                _geo.rel_slope.data.fill_(new_slope)
                # 验证: 计算校准后的 reliability 统计
                rel_all = torch.sigmoid(new_slope * (max_sim_all - new_thr))
                print(f">>> [UncertaintyIsolation] Threshold calibrated: "
                      f"thr={new_thr:.4f}, slope={new_slope:.1f}")
                print(f"    max_sim: p1={p1:.4f}, p10={p10:.4f}, "
                      f"median={med:.4f}, "
                      f"p95={np.percentile(max_sim_all.numpy(), 95):.4f}")
                print(f"    reliability: min={rel_all.min().item():.4f}, "
                      f"mean={rel_all.mean().item():.4f}, "
                      f"p5={np.percentile(rel_all.numpy(), 5):.4f}")

            print(f">>> [GeoHCAN] K-Means init done: "
                  f"{len(all_proj)} samples → {n_centroids} centroids "
                  f"(backbone={self.args.model})")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # [GeoHCAN] K-Means 质心初始化 (纯净基准线模式下跳过)
        if not getattr(self.args, 'no_geohcan', False):
            self.init_geo_centroids(train_loader)
            # 重置 RNG, 消除 K-Means init 对训练随机性的干扰
            # 这样 GeoHCAN 模式和 baseline 从相同 RNG 状态开始训练
            import random
            torch.manual_seed(2021)
            torch.cuda.manual_seed_all(2021)
            np.random.seed(2021)
            random.seed(2021)

            # [GeoHCAN] 冻结几何路由空间, 保留修正 MLP + 门控可训练
            # 冻结: proj, proj_norm, centroids (路由空间, 防崩溃)
            # 可训练: correction_in/out, gate_alpha, routing_log_temp, rel_*
            if getattr(self.args, 'freeze_geo', False) and hasattr(model_ref.backbone, 'geo_hcan'):
                _geo = model_ref.backbone.geo_hcan
                keep_trainable = {
                    'correction',        # correction_in + correction_out
                    'gate_alpha',        # 全局门控
                    'routing_log_temp',  # 路由温度
                    'rel_threshold',     # 可靠性阈值
                    'rel_slope',         # 可靠性斜率
                }
                for name, param in _geo.named_parameters():
                    if not any(k in name for k in keep_trainable):
                        param.requires_grad = False
                frozen_n = sum(p.numel() for p in _geo.parameters()
                               if not p.requires_grad)
                train_n = sum(p.numel() for p in _geo.parameters()
                              if p.requires_grad)
                print(f">>> [GeoHCAN-MoE] Frozen routing: {frozen_n:,} params | "
                      f"Trainable (correction+gate): {train_n:,} params")

        # [freeze_backbone] 冻结骨干, 只训练 GeoHCAN + Head
        if getattr(self.args, 'freeze_backbone', False):
            for param in model_ref.backbone.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f">>> [freeze_backbone] Backbone frozen. "
                  f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # [GeoHCAN] 独立损失权重
        lambda_ortho = self.args.lambda_ortho
        lambda_geo = getattr(self.args, 'lambda_geo', 0.0)
        lambda_latent = getattr(self.args, 'lambda_latent', 0.05)
        geo_warmup_epochs = 2  # L_geo 前 2 个 epoch 不启用

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                # 提前计算 y_true 用于 GeoHCAN 辅助 loss
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_cropped = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # [GeoHCAN] Warmup: L_geo 前 N epoch 不传 y_true
                y_input = batch_y_cropped if (lambda_geo > 0 and epoch >= geo_warmup_epochs) else None

                # 统一模型前向: 训练时返回 (output, ortho_loss, geo_loss, latent_loss)
                model_out = self.model(
                    batch_x, batch_x_mark.float().to(self.device),
                    dec_inp, batch_y_mark.float().to(self.device),
                    y_true=y_input
                )

                # 处理返回值
                if isinstance(model_out, tuple):
                    outputs, ortho_loss, geo_loss, latent_loss = model_out
                else:
                    outputs = model_out
                    ortho_loss = torch.tensor(0.0, device=self.device)
                    geo_loss = torch.tensor(0.0, device=self.device)
                    latent_loss = torch.tensor(0.0, device=self.device)

                outputs = outputs[:, :, f_dim:]

                # MSE 主损失
                loss_mse = criterion(outputs, batch_y_cropped)

                # 总损失 = MSE + λ*Ortho + λ*Geo + λ*Latent
                loss = (loss_mse
                        + lambda_ortho * ortho_loss
                        + lambda_geo * geo_loss
                        + lambda_latent * latent_loss)

                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
                epoch + 1, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop: break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)
                outputs = self.model(
                    batch_x, batch_x_mark.float().to(self.device),
                    dec_inp, batch_y_mark.float().to(self.device)
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                total_loss.append(
                    criterion(
                        outputs[:, :, f_dim:].detach().cpu(),
                        batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu()
                    )
                )
        self.model.train()
        return np.average(total_loss)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)
                outputs = self.model(
                    batch_x, batch_x_mark.float().to(self.device),
                    dec_inp, batch_y_mark.float().to(self.device)
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                preds.append(outputs[:, :, f_dim:].detach().cpu().numpy())
                trues.append(batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe, *ignored = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # ========== 保存结果 & 可视化 ==========
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        self.visualize(preds, trues, folder_path, setting)
        return mae, mse

    def visualize(self, preds, trues, folder_path, setting, num_samples=3, num_features=3):
        """可视化预测结果与真实值对比"""
        N, L, D = preds.shape
        num_samples = min(num_samples, N)
        num_features = min(num_features, D)

        # 1. 单样本多特征对比图
        fig, axes = plt.subplots(num_samples, num_features,
                                 figsize=(5 * num_features, 4 * num_samples))
        if num_samples == 1: axes = axes.reshape(1, -1)
        if num_features == 1: axes = axes.reshape(-1, 1)

        for i in range(num_samples):
            for j in range(num_features):
                ax = axes[i, j]
                ax.plot(trues[i, :, j], label='Ground Truth', color='blue', linewidth=1.5)
                ax.plot(preds[i, :, j], label='Prediction', color='red',
                        linestyle='--', linewidth=1.5)
                ax.set_title(f'Sample {i + 1}, Feature {j + 1}')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(folder_path + 'comparison_multi.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. 单特征聚合图
        fig, ax = plt.subplots(figsize=(12, 6))
        feat_idx = 0
        for i in range(min(5, N)):
            alpha = 0.3 if i > 0 else 1.0
            if i == 0:
                ax.plot(trues[i, :, feat_idx], label='Ground Truth', color='blue',
                        linewidth=2, alpha=alpha)
                ax.plot(preds[i, :, feat_idx], label='Prediction', color='red',
                        linestyle='--', linewidth=2, alpha=alpha)
            else:
                ax.plot(trues[i, :, feat_idx], color='blue', linewidth=1, alpha=alpha)
                ax.plot(preds[i, :, feat_idx], color='red', linestyle='--',
                        linewidth=1, alpha=alpha)
        ax.set_title(f'Feature 1 - Multiple Samples Overlay\n{setting}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(folder_path + 'comparison_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. 误差分布图
        errors = preds - trues
        mean_error = errors.mean(axis=(0, 2))
        std_error = errors.std(axis=(0, 2))
        fig, ax = plt.subplots(figsize=(12, 5))
        time_steps = np.arange(L)
        ax.plot(time_steps, mean_error, color='red', label='Mean Error')
        ax.fill_between(time_steps, mean_error - std_error, mean_error + std_error,
                         color='red', alpha=0.2, label='±1 Std')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f'Prediction Error Distribution Over Time\n{setting}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Error (Pred - True)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(folder_path + 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f">>> Visualization saved to {folder_path}")

    # =========================================================================
    #  预测
    # =========================================================================
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros(
                    [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                ).float().to(self.device)
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :].float().to(self.device), dec_inp],
                    dim=1
                )

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[:, :, f_dim:].detach().cpu().numpy()
                preds.append(pred)

        preds = np.concatenate(preds, axis=0)

        # 反归一化
        if pred_data.scale:
            B, L, D = preds.shape
            preds = preds.reshape(-1, D)
            preds = pred_data.inverse_transform(preds)
            preds = preds.reshape(B, L, D)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)

        n_features = preds[0].shape[-1]
        if hasattr(pred_data, 'cols') and pred_data.cols is not None:
            col_names = (pred_data.cols[-n_features:]
                         if len(pred_data.cols) >= n_features
                         else [f'feat_{i}' for i in range(n_features)])
        else:
            col_names = [f'feat_{i}' for i in range(n_features)]
        pd.DataFrame(preds[0], columns=col_names).to_csv(
            folder_path + 'real_prediction.csv', index=False
        )
        print(">>> Prediction saved to results folder.")
        return
