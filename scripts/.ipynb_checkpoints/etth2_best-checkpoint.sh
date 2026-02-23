#!/bin/bash
# ============================================
# ETTh2 各 pred_len 最优配置汇总
# ============================================

# --- pred_len=96 | Best MSE=1.535 (v2: lr=1e-5, dropout=0.1) ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.1 --learning_rate 0.00001 \
  --train_epochs 6 --patience 3 --batch_size 32 \
  --itr 1 --des 'etth2_best_96' --model_id etth2_96

# --- pred_len=192 | Best MSE=3.589 (v7: lr=1e-5, dropout=0.1, short config) ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 192 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.1 --learning_rate 0.00001 \
  --train_epochs 6 --patience 3 --batch_size 32 \
  --itr 1 --des 'etth2_best_192' --model_id etth2_192

# --- pred_len=336 | Best MSE=3.180 (v1: lr=1e-4, dropout=0.05) ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.05 --learning_rate 0.0001 \
  --train_epochs 12 --patience 5 --batch_size 32 \
  --itr 1 --des 'etth2_best_336' --model_id etth2_336

# --- pred_len=720 | Best MSE=2.526 (v2: lr=1e-4, dropout=0.1) ---
python -u run.py \
  --is_training 1 --model Informer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.1 --learning_rate 0.0001 \
  --train_epochs 12 --patience 5 --batch_size 32 \
  --itr 1 --des 'etth2_best_720' --model_id etth2_720
