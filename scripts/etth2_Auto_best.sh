#!/bin/bash
# ============================================================
# ETTh2 Autoformer + GeoHCAN 各 pred_len 最优配置汇总
# ============================================================
# 基于 ETTh1 最优配置迁移，针对 ETTh2 数据特点微调
# ETTh2 波动较大，适当调整 gate_power 和 learning_rate
# ============================================================

# --- pred_len=96 | MSE=0.3479 | gate_power=0, num_fine=8 ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 8 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 0 \
  --dropout 0.05 --learning_rate 0.0005 \
  --train_epochs 10 --patience 5 --batch_size 32 \
  --itr 1 --des 'Auto_best_96' --model_id Auto_best_ETTh2_96

# --- pred_len=192 | MSE=0.4435 | gate_power=2, num_fine=8 ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 192 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 8 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.05 --learning_rate 0.0005 \
  --train_epochs 10 --patience 5 --batch_size 32 \
  --itr 1 --des 'Auto_best_192' --model_id Auto_best_ETTh2_192

# --- pred_len=336 | MSE=0.4665 | gate_power=2, num_fine=8 ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 8 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 2 \
  --dropout 0.05 --learning_rate 0.0005 \
  --train_epochs 10 --patience 5 --batch_size 32 \
  --itr 1 --des 'Auto_best_336' --model_id Auto_best_ETTh2_336

# --- pred_len=720 | MSE=0.4915 | gate_power=4, num_fine=8 ---
python -u run.py \
  --is_training 1 --model Autoformer --data ETTh2 \
  --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --moving_avg 25 --num_fine 8 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 --gate_power 4 \
  --dropout 0.05 --learning_rate 0.0005 \
  --train_epochs 10 --patience 5 --batch_size 32 \
  --itr 1 --des 'Auto_best_720' --model_id Auto_best_ETTh2_720