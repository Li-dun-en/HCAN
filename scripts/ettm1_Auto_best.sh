#!/bin/bash
# ============================================================
# ETTm1 Autoformer + GeoHCAN 各 pred_len 最优配置汇总
# ============================================================
# 基于 ETTh2 最优配置迁移，针对 ETTm1 数据特点微调
# ETTm1 为分钟级数据，粒度更细，适当调整参数
# ============================================================

# --- pred_len=96 | MSE=xxx | gate_power=1, num_fine=8 ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 8 --d_proj 16 \
  --lambda_geo 0.5 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 1 --gate_power 1 \
  --dropout 0.05 --learning_rate 0.0005 \
  --train_epochs 10 --patience 5 --batch_size 32 \
  --itr 1 --des 'Auto_best_96' --model_id Auto_best_ETTm1_96

# --- pred_len=192 | MSE=xxx | gate_power=2, num_fine=8 ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 192 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 10 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.002 --lambda_reg 0.05 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 1 --gate_power 2 \
  --dropout 0.06 --learning_rate 0.0004 \
  --train_epochs 10 --patience 6 --batch_size 32 \
  --itr 1 --des 'Auto_best_192' --model_id Auto_best_ETTm1_192

# --- pred_len=336 | MSE=xxx ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 12 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.002 --lambda_reg 0.05 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 1 --gate_power 2 \
  --dropout 0.06 --learning_rate 0.0004 \
  --train_epochs 10 --patience 8 --batch_size 32 \
  --itr 1 --des 'Auto_best_336' --model_id Auto_best_ETTm1_336

# --- pred_len=720 | MSE=xxx ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 16 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.005 --lambda_reg 0.1 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 1 --gate_power 2 \
  --dropout 0.1 --learning_rate 0.0003 \
  --train_epochs 10 --patience 5 --batch_size 32 \
  --itr 1 --des 'Auto_best_720' --model_id Auto_best_ETTm1_720
