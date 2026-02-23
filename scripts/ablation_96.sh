#!/bin/bash
# =============================================================
# 系统消融测试: 找出 pred_len=96 从 0.305 退化的根因
# 
# 变量:
#   A) lr=1e-5 vs lr=1e-4
#   B) factor=3 vs factor=5
#   C) use_energy_gate=1 vs 0
#   D) use_proj_ln / detach_temp
#
# 策略: 先测超参数差异 (A+B), 再测开关组合 (C+D)
# =============================================================

BASE="python -u run.py --is_training 1 --model Informer --data custom \
  --root_path ./dataset/weather/ --data_path weather.csv --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --num_fine 4 --d_proj 64 \
  --lambda_geo 0.0 --lambda_latent 0.0 \
  --train_epochs 6 --batch_size 32 --itr 1"

echo "============================================"
echo "Test 1: 还原原始超参 lr=1e-5, factor=3, 无保护, 有energy_gate"
echo "  → 最接近 0.305 的配置 (除了 K-Means init 和 T mapping)"
echo "============================================"
$BASE --factor 3 --learning_rate 0.00001 \
  --use_proj_ln 0 --detach_temp 0 --use_energy_gate 1 \
  --patience 3 --des 'ablation_A' --model_id abl_lr1e5_f3_EG

echo ""
echo "============================================"
echo "Test 2: 只改 lr → 1e-4 (其他同 Test 1)"
echo "  → 隔离 learning_rate 的影响"
echo "============================================"
$BASE --factor 3 --learning_rate 0.0001 \
  --use_proj_ln 0 --detach_temp 0 --use_energy_gate 1 \
  --patience 3 --des 'ablation_B' --model_id abl_lr1e4_f3_EG

echo ""
echo "============================================"
echo "Test 3: 只改 factor → 5 (其他同 Test 1)"
echo "  → 隔离 factor 的影响"
echo "============================================"
$BASE --factor 5 --learning_rate 0.00001 \
  --use_proj_ln 0 --detach_temp 0 --use_energy_gate 1 \
  --patience 3 --des 'ablation_C' --model_id abl_lr1e5_f5_EG

echo ""
echo "============================================"
echo "Test 4: lr=1e-5, factor=3, 无 energy_gate"
echo "  → 隔离 energy_gate 的影响"
echo "============================================"
$BASE --factor 3 --learning_rate 0.00001 \
  --use_proj_ln 0 --detach_temp 0 --use_energy_gate 0 \
  --patience 3 --des 'ablation_D' --model_id abl_lr1e5_f3_noEG

echo ""
echo "============================================"
echo "所有消融测试完成! 汇总结果:"
echo "============================================"
echo "Test 1 (lr=1e-5, factor=3, EG=1): 最接近原始0.305"
echo "Test 2 (lr=1e-4, factor=3, EG=1): 隔离lr"
echo "Test 3 (lr=1e-5, factor=5, EG=1): 隔离factor"
echo "Test 4 (lr=1e-5, factor=3, EG=0): 隔离energy_gate"
