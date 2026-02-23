from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer 
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
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_instance = Informer.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model_instance = nn.DataParallel(model_instance, device_ids=self.args.device_ids)
        return model_instance

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    # =========================================================================
    # [SP-DEFI] Sorted Projection K-Means 初始化
    # =========================================================================
    def init_defi_centroids_sorted(self, train_loader):
        """
        基于 Decoder 预测区间特征进行聚类，并按 Y 值均值排序 Centroids
        """
        print(">>> [SP-DEFI] Initializing Centroids with Sorted Projection...")
        self.model.eval()
        device = self.device
        
        proj_features = []
        norm_y_values = []
        limit = 100 
        
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if i >= limit: break
                
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(device)
                
                # 1. Encoder
                enc_out = model_ref.enc_embedding(batch_x, batch_x_mark)
                enc_out, _ = model_ref.encoder(enc_out, attn_mask=None)
                
                # 2. Decoder
                dec_out = model_ref.dec_embedding(dec_inp, batch_y_mark)
                dec_out = model_ref.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
                
                # 3. 截取预测区间
                dec_feat = dec_out[:, -self.args.pred_len:, :]
                
                # 4. 模拟 defi_layer 内部 RevIN + Projection
                mean = dec_feat.mean(dim=1, keepdim=True)
                std = dec_feat.std(dim=1, keepdim=True) + 1e-5
                f_norm = (dec_feat - mean) / std
                f_proj = model_ref.defi_layer.proj_in(f_norm)
                
                # 5. 获取对应的相对标签用于排序
                true_y = batch_y[:, -self.args.pred_len:, :]
                y_mean = true_y.mean(dim=1, keepdim=True)
                y_std = true_y.std(dim=1, keepdim=True) + 1e-5
                y_norm = (true_y - y_mean) / y_std
                
                proj_features.append(f_proj.reshape(-1, f_proj.shape[-1]).cpu().numpy())
                norm_y_values.append(y_norm.reshape(-1).cpu().numpy())
                
        all_proj = np.concatenate(proj_features, axis=0)
        all_vals = np.concatenate(norm_y_values, axis=0)
        
        if all_proj.shape[0] > 50000:
            idx = np.random.choice(all_proj.shape[0], 50000, replace=False)
            all_proj = all_proj[idx]
            all_vals = all_vals[idx]

        def cluster_and_sort(num_clusters, target_param, anchor_buffer=None):
            print(f"    Running K-Means (K={num_clusters})...")
            kmeans = KMeans(n_clusters=num_clusters, n_init=10).fit(all_proj)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # 计算每个 Cluster 对应的 Y 均值
            cluster_means = [all_vals[labels == k].mean() if (labels == k).sum() > 0 else 0 for k in range(num_clusters)]
            # 按均值从小到大排序 (Index 0 = Valley, Index K-1 = Peak)
            sort_idx = np.argsort(cluster_means)
            sorted_means = np.array(cluster_means)[sort_idx]
            
            # 保存排序后的 centroids
            target_param.data = torch.tensor(centers[sort_idx]).float().to(device)
            
            # [新增] 保存 anchor_means buffer (用于 Loss 计算)
            if anchor_buffer is not None:
                anchor_buffer.data = torch.tensor(sorted_means).float().to(device)

        # Fine-grained centroids + anchor_means
        cluster_and_sort(
            model_ref.defi_layer.num_classes_fine, 
            model_ref.defi_layer.centroids_fine,
            model_ref.defi_layer.anchor_means  # [新增] 保存真实聚类均值
        )
        # Coarse-grained centroids (无需 anchor_means)
        cluster_and_sort(model_ref.defi_layer.num_classes_coarse, model_ref.defi_layer.centroids_coarse)
        print(">>> [SP-DEFI+] Initialization Complete.")

    def calc_sp_defi_loss(self, belief_f, raw_offsets, true_y, rev_stats=None):
        """
        SP-DEFI+ 辅助损失
        """
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        valid_len = min(96, self.args.pred_len)
        
        # 1. Myopic Mask
        if valid_len <= 0:
            return torch.tensor(0.0, device=true_y.device)
        
        belief_f = belief_f[:, :valid_len, :]
        raw_offsets = raw_offsets[:, :valid_len, :]
        true_y_crop = true_y[:, :valid_len, :]
        
        # 2. [Fix] 维度匹配修正
        # rev_stats 来自 d_model (512维), true_y 是 c_out (21维), 无法直接运算。
        # 这里必须使用 Label 自身的 Instance Normalization。
        y_mean = true_y_crop.mean(dim=1, keepdim=True)
        y_std = true_y_crop.std(dim=1, keepdim=True) + 1e-5
        y_norm = (true_y_crop - y_mean) / y_std
        
        # 3. 使用真实聚类均值作为锚点
        anchors = model_ref.defi_layer.anchor_means  # [K]
        
        # 4. 动态软目标生成
        y_scalar = y_norm.mean(dim=-1, keepdim=True)  # [B, L, 1]
        dist_sq = (y_scalar - anchors.view(1, 1, -1)) ** 2  # [B, L, K]
        rbf_gamma = getattr(self.args, 'rbf_gamma', 1.0)
        q = F.softmax(-dist_sq / (2 * rbf_gamma ** 2), dim=-1)
        
        # 5. KL 散度分类损失
        loss_cls = F.kl_div(torch.log(belief_f + 1e-8), q, reduction='batchmean')
        
        # 6. 软回归损失
        true_offset = y_scalar - anchors.view(1, 1, -1)
        loss_reg = (q * (raw_offsets - true_offset) ** 2).sum(dim=-1).mean()
        
        lambda_cls = getattr(self.args, 'lambda_cls', 0.1)
        lambda_reg = getattr(self.args, 'lambda_reg', 0.05)
        
        return lambda_cls * loss_cls + lambda_reg * loss_reg

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        self.init_defi_centroids_sorted(train_loader) #

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # [重要] 模型返回 5 个值: output, belief_fine, belief_coarse, offsets, rev_stats
                outputs, b_fine, b_coarse, offsets, rev_stats = self.model(
                    batch_x, batch_x_mark.float().to(self.device), 
                    dec_inp, batch_y_mark.float().to(self.device)
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y_cropped = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # MSE 主损失
                loss_mse = criterion(outputs, batch_y_cropped)
                
                # SP-DEFI+ 辅助损失 (传入 rev_stats 确保归一化对齐)
                loss_aux = self.calc_sp_defi_loss(b_fine, offsets, batch_y_cropped, rev_stats)
                loss = loss_mse + loss_aux
                
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch + 1, train_loss, vali_loss))
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
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark.float().to(self.device), dec_inp, batch_y_mark.float().to(self.device))
                f_dim = -1 if self.args.features == 'MS' else 0
                total_loss.append(criterion(outputs[:, :, f_dim:].detach().cpu(), batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu()))
        self.model.train()
        return np.average(total_loss)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test: self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark.float().to(self.device), dec_inp, batch_y_mark.float().to(self.device))
                f_dim = -1 if self.args.features == 'MS' else 0
                preds.append(outputs[:, :, f_dim:].detach().cpu().numpy())
                trues.append(batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy())
        preds, trues = np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe, *ignored = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        return mae, mse

    # =========================================================================
    # [补全] 真实预测逻辑 (Predict)
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

                # 构造 decoder 输入 (前缀为 label_len)
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :].float().to(self.device), dec_inp], dim=1)

                # 推理时模型仅返回预测值 (output)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 截取预测部分并转为 NumPy
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[:, :, f_dim:].detach().cpu().numpy()
                preds.append(pred)

        preds = np.concatenate(preds, axis=0)
        
        # 反归一化 (Inverse Transform)
        if pred_data.scale:
            # 数据形状对齐处理 [N, L, D] -> [N*L, D]
            B, L, D = preds.shape
            preds = preds.reshape(-1, D)
            preds = pred_data.inverse_transform(preds)
            preds = preds.reshape(B, L, D)

        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        
        # 保存 CSV 结果 (取第一个样本展示)
        pd.DataFrame(preds[0], columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)
        print(">>> Prediction saved to results folder.")
        return
