#!/bin/bash
# TimesNet + GeoHCAN 消融实验
# 目标: 验证 "活的 GeoHCAN (Exp B) + 稳定训练 (Exp D)" 能否突破 baseline 0.389
# 基础: Exp B (use_proj_ln=0, detach_temp=0) × Exp D (lambda_latent=0.01)

cd /root/autodl-tmp/HCAN-main/HCAN-main
mkdir -p logs

BASE="--is_training 1 --model TimesNet --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 16 --d_ff 32 --e_layers 2 \
  --dropout 0.1 --top_k 5 --num_kernels 6 \
  --batch_size 32 --geohcan_d_model 512 \
  --learning_rate 0.0001 --lradj type1 \
  --lambda_geo 0.0"

echo "=========================================="
echo "Abl-1: B+D 组合 (活GeoHCAN + 温和锚定)"
echo "  use_proj_ln=0, detach_temp=0, lambda_latent=0.01"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 15 --patience 5 \
  --d_proj 16 --lambda_latent 0.01 \
  --use_proj_ln 0 --detach_temp 0 \
  --itr 1 --des 'abl1_BD' --model_id abl1_BD 2>&1 | tee logs/abl1_BD.log

echo ""
echo "=========================================="
echo "Abl-2: B+D+C 组合 (活GeoHCAN + 温和锚定 + 8质心)"
echo "  use_proj_ln=0, detach_temp=0, lambda_latent=0.01, num_fine=8"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 15 --patience 5 \
  --d_proj 16 --lambda_latent 0.01 \
  --use_proj_ln 0 --detach_temp 0 \
  --num_fine 8 \
  --itr 1 --des 'abl2_BDC' --model_id abl2_BDC 2>&1 | tee logs/abl2_BDC.log

echo ""
echo "=========================================="
echo "Abl-3: B+D + d_proj=32 (活GeoHCAN + 温和锚定 + 宽投影)"
echo "  use_proj_ln=0, detach_temp=0, lambda_latent=0.01, d_proj=32"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 15 --patience 5 \
  --d_proj 32 --lambda_latent 0.01 \
  --use_proj_ln 0 --detach_temp 0 \
  --itr 1 --des 'abl3_BD_dp32' --model_id abl3_BD_dp32 2>&1 | tee logs/abl3_BD_dp32.log

echo ""
echo "=========================================="
echo "Abl-4: B+D + energy_gate=1 (活GeoHCAN + 温和锚定 + 能量门控)"
echo "  use_proj_ln=0, detach_temp=0, lambda_latent=0.01, energy_gate=1"
echo "=========================================="
python -u run.py $BASE \
  --train_epochs 15 --patience 5 \
  --d_proj 16 --lambda_latent 0.01 \
  --use_proj_ln 0 --detach_temp 0 \
  --use_energy_gate 1 \
  --itr 1 --des 'abl4_BD_egate' --model_id abl4_BD_egate 2>&1 | tee logs/abl4_BD_egate.log

echo ""
echo "=========================================="
echo "=== 消融实验完成, 汇总 ==="
echo "=========================================="
echo "Baseline (纯 TimesNet):         MSE=0.3890"
echo "Exp B (no_proj_ln+no_detach):   MSE=0.3924"
echo "Exp D (lambda_latent=0.01):     MSE=0.3900"
echo ""
echo "--- Abl-1 (B+D): ---"
grep "^mse:" logs/abl1_BD.log
echo "--- Abl-2 (B+D+C, nf=8): ---"
grep "^mse:" logs/abl2_BDC.log
echo "--- Abl-3 (B+D, d_proj=32): ---"
grep "^mse:" logs/abl3_BD_dp32.log
echo "--- Abl-4 (B+D, energy_gate): ---"
grep "^mse:" logs/abl4_BD_egate.log
