#!/bin/bash
# ============================================================
# ETTm1 最优指令汇总 (各 pred_len 历史最佳 MSE)
# ============================================================
# pred_len |  MSE   |  MAE   | 版本 | lr schedule
# ---------|--------|--------|------|------------
#    96    | 0.482  | 0.482  |  v2  | type1
#   192    | 0.588  | 0.558  |  v6  | lradj=6
#   336    | 0.673  | 0.625  |  v6  | lradj=6
#   720    | 1.040  | 0.767  | v4b  | type1
# ---------|--------|--------|------|------------
#   平均   | 0.696  | 0.608  |      |
# ============================================================

# ===== pred_len=96  (MSE=0.482, v2, lradj=type1) =====
python -u run.py --is_training 1 --model Informer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 0 --use_energy_gate 0 \
  --dropout 0.1 \
  --train_epochs 10 --batch_size 32 --learning_rate 0.00005 \
  --patience 5 --des 'ettm1_v2' --itr 1 --model_id ettm1_96 && \

# ===== pred_len=192  (MSE=0.588, v6, lradj=6) =====
python -u run.py --is_training 1 --model Informer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --features M \
  --seq_len 96 --label_len 48 --pred_len 192 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --num_fine 4 --d_proj 16 \
  --lambda_cls 0.1 --lambda_reg 0.05 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 0 --use_energy_gate 0 \
  --dropout 0.1 \
  --train_epochs 15 --batch_size 32 --learning_rate 0.00002 \
  --lradj '6' \
  --patience 7 --des 'ettm1_v6' --itr 1 --model_id ettm1_192 && \

# ===== pred_len=336  (MSE=0.673, v6, lradj=6) =====
python -u run.py --is_training 1 --model Informer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --features M \
  --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 3 --num_fine 4 --d_proj 32 \
  --lambda_cls 0.05 --lambda_reg 0.02 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --dropout 0.15 \
  --train_epochs 15 --batch_size 64 --learning_rate 0.00002 \
  --lradj '6' \
  --patience 7 --des 'ettm1_v6' --itr 1 --model_id ettm1_336 && \

# ===== pred_len=720  (MSE=1.040, v4b, lradj=type1) =====
python -u run.py --is_training 1 --model Informer --data ETTm1 \
  --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --features M \
  --seq_len 96 --label_len 48 --pred_len 720 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 3 --num_fine 4 --d_proj 16 \
  --lambda_cls 0.05 --lambda_reg 0.02 \
  --lambda_geo 0.0 --lambda_latent 0.001 \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --dropout 0.12 \
  --train_epochs 10 --batch_size 64 --learning_rate 0.0003 \
  --patience 5 --des 'ettm1_v4b' --itr 1 --model_id ettm1_720
