#!/bin/bash
# ================================================================
# FEDformer + GeoHCAN 调参 v2 — 保守策略
# ================================================================
# v1 教训: lr=5e-4 + nf=8 全部过拟合 (96: 0.395~0.401 vs 默认 0.375)
#   原因: FEDformer 频域特征比 Autoformer 更敏感, 高 lr 破坏频域结构
#
# v2 策略: 锁定 lr=1e-4, 围绕已验证配置做微调
#   默认基线: 96=0.375  192=0.419  336=0.456  720=0.461
#   纯净基线: 96=0.376  192=0.420  336=0.459  720=0.481
# ================================================================

cd /root/autodl-tmp/HCAN-main/HCAN-main

LOG_DIR="logs/fedformer_tune_v2"
mkdir -p $LOG_DIR

COMMON="--is_training 1 --model FEDformer --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 \
  --factor 5 --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --lambda_geo 0.0 --dropout 0.05 --batch_size 32 --itr 1"

# ================================================================
#  pred_len = 96   (默认: nf=4, dp=16, gp=2, ll=0.001, lr=1e-4 → 0.375)
# ================================================================
PL=96
echo "==================== pred_len=$PL ===================="

echo "--- 96-A: nf=6, gp=2 (适度增加质心) ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 6 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_96A' --model_id fed_v2_96A 2>&1 | tee $LOG_DIR/96_A.log

echo "--- 96-B: nf=4, gp=0 (关闭gate) ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 0 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_96B' --model_id fed_v2_96B 2>&1 | tee $LOG_DIR/96_B.log

echo "--- 96-C: nf=4, d_proj=32 (更宽投影) ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 32 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_96C' --model_id fed_v2_96C 2>&1 | tee $LOG_DIR/96_C.log

echo "--- 96-D: nf=4, lambda_latent=0.005 (加强锚定) ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 2 --lambda_latent 0.005 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_96D' --model_id fed_v2_96D 2>&1 | tee $LOG_DIR/96_D.log

echo "--- 96-E: nf=8, lr=5e-5 (多质心+低lr防过拟合) ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 8 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.00005 --train_epochs 15 --patience 5 \
  --des 'fed_v2_96E' --model_id fed_v2_96E 2>&1 | tee $LOG_DIR/96_E.log

# ================================================================
#  pred_len = 192   (默认: nf=4, dp=16, gp=2, ll=0.001, lr=1e-4 → 0.419)
# ================================================================
PL=192
echo "==================== pred_len=$PL ===================="

echo "--- 192-A: nf=6, gp=2 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 6 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_192A' --model_id fed_v2_192A 2>&1 | tee $LOG_DIR/192_A.log

echo "--- 192-B: nf=4, gp=0 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 0 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_192B' --model_id fed_v2_192B 2>&1 | tee $LOG_DIR/192_B.log

echo "--- 192-C: nf=4, d_proj=32 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 32 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_192C' --model_id fed_v2_192C 2>&1 | tee $LOG_DIR/192_C.log

echo "--- 192-D: nf=4, lambda_latent=0.005 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 2 --lambda_latent 0.005 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_192D' --model_id fed_v2_192D 2>&1 | tee $LOG_DIR/192_D.log

echo "--- 192-E: nf=8, lr=5e-5 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 8 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.00005 --train_epochs 15 --patience 5 \
  --des 'fed_v2_192E' --model_id fed_v2_192E 2>&1 | tee $LOG_DIR/192_E.log

# ================================================================
#  pred_len = 336   (默认: nf=4, dp=16, gp=2, ll=0.001, lr=1e-4 → 0.456)
# ================================================================
PL=336
echo "==================== pred_len=$PL ===================="

echo "--- 336-A: nf=6, gp=2 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 6 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_336A' --model_id fed_v2_336A 2>&1 | tee $LOG_DIR/336_A.log

echo "--- 336-B: nf=4, gp=4 (长程加大gate) ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 4 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_336B' --model_id fed_v2_336B 2>&1 | tee $LOG_DIR/336_B.log

echo "--- 336-C: nf=4, d_proj=32 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 32 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_336C' --model_id fed_v2_336C 2>&1 | tee $LOG_DIR/336_C.log

echo "--- 336-D: nf=4, lambda_latent=0.005 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 2 --lambda_latent 0.005 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_336D' --model_id fed_v2_336D 2>&1 | tee $LOG_DIR/336_D.log

echo "--- 336-E: nf=8, lr=5e-5 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 8 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.00005 --train_epochs 15 --patience 5 \
  --des 'fed_v2_336E' --model_id fed_v2_336E 2>&1 | tee $LOG_DIR/336_E.log

# ================================================================
#  pred_len = 720   (默认: nf=4, dp=16, gp=2, ll=0.001, lr=5e-4 → 0.461)
#  注: 720 的默认 lr 已经是 5e-4, 但降到 1e-4 可能更稳
# ================================================================
PL=720
echo "==================== pred_len=$PL ===================="

echo "--- 720-A: nf=6, gp=2, lr=1e-4 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 6 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_720A' --model_id fed_v2_720A 2>&1 | tee $LOG_DIR/720_A.log

echo "--- 720-B: nf=4, gp=4, lr=1e-4 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 4 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_720B' --model_id fed_v2_720B 2>&1 | tee $LOG_DIR/720_B.log

echo "--- 720-C: nf=4, d_proj=32, lr=1e-4 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 32 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_720C' --model_id fed_v2_720C 2>&1 | tee $LOG_DIR/720_C.log

echo "--- 720-D: nf=4, lambda_latent=0.005, lr=1e-4 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 4 --d_proj 16 --gate_power 2 --lambda_latent 0.005 \
  --learning_rate 0.0001 --train_epochs 10 --patience 3 \
  --des 'fed_v2_720D' --model_id fed_v2_720D 2>&1 | tee $LOG_DIR/720_D.log

echo "--- 720-E: nf=8, lr=5e-5 ---"
python -u run.py $COMMON --pred_len $PL \
  --num_fine 8 --d_proj 16 --gate_power 2 --lambda_latent 0.001 \
  --learning_rate 0.00005 --train_epochs 15 --patience 5 \
  --des 'fed_v2_720E' --model_id fed_v2_720E 2>&1 | tee $LOG_DIR/720_E.log

# ================================================================
#  汇总结果
# ================================================================
echo ""
echo "================================================================"
echo "  FEDformer + GeoHCAN 调参 v2 结果汇总"
echo "================================================================"
echo "纯净基线:  96=0.376  192=0.420  336=0.459  720=0.481"
echo "默认配置:  96=0.375  192=0.419  336=0.456  720=0.461"
echo "----------------------------------------------------------------"

for PL in 96 192 336 720; do
  echo ""
  echo "--- pred_len=$PL ---"
  for EXP in A B C D E; do
    LOG="$LOG_DIR/${PL}_${EXP}.log"
    if [ -f "$LOG" ]; then
      MSE=$(grep "^mse:" "$LOG" | tail -1 | awk -F'[:,]' '{printf "%.4f", $2}')
      MAE=$(grep "^mse:" "$LOG" | tail -1 | awk -F'[:,]' '{printf "%.4f", $4}')
      DESC=$(grep "^---" "$LOG" | head -1 | sed 's/^--- //' | sed 's/ ---$//')
      echo "  ${PL}-${EXP}: MSE=${MSE} MAE=${MAE}  ($DESC)"
    fi
  done
done

echo ""
echo "================================================================"
echo "  调参完成!"
echo "================================================================"
