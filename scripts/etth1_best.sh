#!/bin/bash
# ============================================
# ETTh1 各 pred_len 最优配置汇总 (v5: inject_gate)
# ============================================

# --- pred_len=96 | Best MSE=0.753 (v5: factor=5, d_proj=16, inject_gate) ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.05 --learning_rate 0.0001 \
  --train_epochs 6 --patience 3 --batch_size 32 \
  --itr 1 --des 'etth1_best_96' --model_id etth1_96

# --- pred_len=192 | v5 配置 ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 192 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.05 --learning_rate 0.0001 \
  --train_epochs 6 --patience 3 --batch_size 32 \
  --itr 1 --des 'etth1_best_192' --model_id etth1_192

# --- pred_len=336 | v5 配置 ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.05 --learning_rate 0.0001 \
  --train_epochs 6 --patience 3 --batch_size 32 \
  --itr 1 --des 'etth1_best_336' --model_id etth1_336

# --- pred_len=720 | v5 配置 (lr=0.0005) ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.05 --learning_rate 0.0005 \
  --train_epochs 6 --patience 3 --batch_size 32 \
  --itr 1 --des 'etth1_best_720' --model_id etth1_720
