#!/bin/bash
# TimesNet + GeoHCAN 调参批量实验
# 基准配置 (MSE=0.393): d_model=16, d_ff=32, geohcan=512, d_proj=16, lambda_latent=0.001, lr=1e-4, type1
# Baseline (纯 TimesNet): MSE=0.389

cd /root/autodl-tmp/HCAN-main/HCAN-main

BASE="--is_training 1 --model TimesNet --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 16 --d_ff 32 --e_layers 2 \
  --dropout 0.1 --top_k 5 --num_kernels 6 \
  --batch_size 32 --geohcan_d_model 512"

echo "=========================================="
echo "Exp A: d_proj=32 (更宽投影空间, 改善质心匹配)"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --lradj type1 \
  --d_proj 32 --lambda_latent 0.001 --lambda_geo 0.0 \
  --itr 1 --des 'tune_A_dproj32' --model_id tune_A 2>&1 | tee logs/tune_A_dproj32.log

echo ""
echo "=========================================="
echo "Exp B: use_proj_ln=0 + detach_temp=0 (关闭长序列保护, 释放GeoHCAN表达力)"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --lradj type1 \
  --d_proj 16 --lambda_latent 0.001 --lambda_geo 0.0 \
  --use_proj_ln 0 --detach_temp 0 \
  --itr 1 --des 'tune_B_noprojln' --model_id tune_B 2>&1 | tee logs/tune_B_noprojln.log

echo ""
echo "=========================================="
echo "Exp C: num_fine=8 (更多质心, 更细粒度几何结构)"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --lradj type1 \
  --d_proj 16 --lambda_latent 0.001 --lambda_geo 0.0 \
  --num_fine 8 \
  --itr 1 --des 'tune_C_nf8' --model_id tune_C 2>&1 | tee logs/tune_C_nf8.log

echo ""
echo "=========================================="
echo "Exp D: lambda_latent=0.01 + lr=1e-4 + epochs=15 (温和锚定 + 更长训练)"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 15 --patience 5 \
  --learning_rate 0.0001 --lradj type1 \
  --d_proj 16 --lambda_latent 0.01 --lambda_geo 0.0 \
  --itr 1 --des 'tune_D_lat01' --model_id tune_D 2>&1 | tee logs/tune_D_lat01.log

echo ""
echo "=========================================="
echo "=== 全部完成, 汇总结果 ==="
echo "=========================================="
echo "Baseline (纯 TimesNet):  MSE=0.389"
echo "当前最佳 (v1):           MSE=0.393"
echo ""
echo "--- Exp A (d_proj=32): ---"
grep "^mse:" logs/tune_A_dproj32.log
echo "--- Exp B (no_proj_ln + no_detach): ---"
grep "^mse:" logs/tune_B_noprojln.log
echo "--- Exp C (num_fine=8): ---"
grep "^mse:" logs/tune_C_nf8.log
echo "--- Exp D (lambda_latent=0.01): ---"
grep "^mse:" logs/tune_D_lat01.log
