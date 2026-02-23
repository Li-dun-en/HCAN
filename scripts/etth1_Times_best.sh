#!/bin/bash
# ============================================================
# ETTh1 TimesNet + GeoHCAN 各 pred_len 最优配置汇总
# ============================================================
# 架构: TimesNet Backbone(d_model=16) → 残差修正模式
#   y_base = base_projection(features)          TimesNet 自身预测
#   geo_in = FusedAdapter(raw=x_norm, feat)     Raw(7→32) + Feat(16→32) 双流融合
#   delta  = Head(GeoHCAN(geo_in)) * res_gate   ReZero 门控修正量
#   output = (y_base + delta) → RevIN denorm
#
# FusedAdapter: T<P 时使用线性插值 (F.interpolate) 代替重复填充
#
# 纯净 TimesNet 基线: 96=0.389, 192=0.453, 336=0.491, 720=0.521 (avg=0.464)
# GeoHCAN 最优:       96=0.390, 192=0.432, 336=0.481, 720=0.472 (avg=0.444)
# 提升:                 ~0%     -4.6%     -2.0%     -9.4%  (avg=-4.3%)
# ============================================================

# --- pred_len=96 | MSE=0.390 | LR=5e-4, lradj=6, nf=8 ---
python -u run.py \
  --is_training 1 --model TimesNet --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 16 --d_ff 32 --e_layers 2 \
  --dropout 0.1 --top_k 5 --num_kernels 6 \
  --geohcan_d_model 32 --d_proj 16 --num_fine 8 \
  --lambda_latent 0.02 --lambda_geo 0.0 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --learning_rate 0.0005 --lradj 6 \
  --train_epochs 10 --patience 3 --batch_size 32 \
  --itr 1 --des 'Times_best_96' --model_id Times_best_96

# --- pred_len=192 | MSE=0.432 | LR=1e-4, lradj=type1, nf=8 ---
python -u run.py \
  --is_training 1 --model TimesNet --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 192 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 16 --d_ff 32 --e_layers 2 \
  --dropout 0.1 --top_k 5 --num_kernels 6 \
  --geohcan_d_model 32 --d_proj 16 --num_fine 8 \
  --lambda_latent 0.005 --lambda_geo 0.0 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --learning_rate 0.0001 --lradj type1 \
  --train_epochs 10 --patience 3 --batch_size 32 \
  --itr 1 --des 'Times_best_192' --model_id Times_best_192

# --- pred_len=336 | MSE=0.481 | LR=2e-4, lradj=type1, nf=8 ---
python -u run.py \
  --is_training 1 --model TimesNet --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 16 --d_ff 32 --e_layers 2 \
  --dropout 0.1 --top_k 5 --num_kernels 6 \
  --geohcan_d_model 32 --d_proj 16 --num_fine 8 \
  --lambda_latent 0.005 --lambda_geo 0.0 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --learning_rate 0.0002 --lradj type1 \
  --train_epochs 10 --patience 3 --batch_size 32 \
  --itr 1 --des 'Times_best_336' --model_id Times_best_336

# --- pred_len=720 | MSE=0.472 | LR=1e-4, lradj=type1, nf=8 ---
python -u run.py \
  --is_training 1 --model TimesNet --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 16 --d_ff 32 --e_layers 2 \
  --dropout 0.1 --top_k 5 --num_kernels 6 \
  --geohcan_d_model 32 --d_proj 16 --num_fine 8 \
  --lambda_latent 0.005 --lambda_geo 0.0 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --learning_rate 0.0001 --lradj type1 \
  --train_epochs 10 --patience 3 --batch_size 32 \
  --itr 1 --des 'Times_best_720' --model_id Times_best_720
