#!/bin/bash
# ================================================================
# Non-stationary Transformer 纯净基准线 (--no_geohcan)
# 超参数来源: TSLib 官方脚本
#   https://github.com/thuml/Time-Series-Library/blob/main/scripts/
#   long_term_forecast/ETT_script/Nonstationary_Transformer_ETTh1.sh
#
# 关键超参 (与 Informer 不同!):
#   d_model=128 (非 512), p_hidden_dims=256 256, factor=3
# ================================================================

# ===================== ETTh1 =====================
for pred_len in 96 192 336 720; do
python -u run.py \
  --is_training 1 --model Nonstationary_Transformer --no_geohcan \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 3 --dropout 0.05 --activation gelu \
  --p_hidden_dims 256 256 --p_hidden_layers 2 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 --batch_size 32 \
  --itr 1 --des "ns_baseline" --model_id "NST_ETTh1_${pred_len}"
done
