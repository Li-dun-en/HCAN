#!/bin/bash
# TimesNet ETTh1 全预测长度: Baseline + GeoHCAN (Exp D config)
# 原论文参考: 96→0.384, 192→0.436, 336→0.491, 720→0.521
# 我们的 96 baseline: 0.389, 96 GeoHCAN: 0.390

cd /root/autodl-tmp/HCAN-main/HCAN-main
mkdir -p logs

COMMON="--is_training 1 --model TimesNet --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 16 --d_ff 32 --e_layers 2 \
  --top_k 5 --num_kernels 6 --batch_size 32"

# ====================================================================
#  PART 1: Baseline (纯 TimesNet, --no_geohcan)
#  配置: d_model=16, d_ff=32, lradj=type1, dropout=0.1
# ====================================================================

echo "============================================"
echo "  Baseline pred_len=192"
echo "============================================"
python -u run.py $COMMON --pred_len 192 \
  --dropout 0.1 --learning_rate 0.0001 --lradj type1 \
  --train_epochs 10 --patience 3 \
  --no_geohcan \
  --itr 1 --des 'timesnet_base_192' --model_id tn_base_192 2>&1 | tee logs/tn_base_192.log

echo "============================================"
echo "  Baseline pred_len=336"
echo "============================================"
python -u run.py $COMMON --pred_len 336 \
  --dropout 0.1 --learning_rate 0.0001 --lradj type1 \
  --train_epochs 10 --patience 3 \
  --no_geohcan \
  --itr 1 --des 'timesnet_base_336' --model_id tn_base_336 2>&1 | tee logs/tn_base_336.log

echo "============================================"
echo "  Baseline pred_len=720"
echo "============================================"
python -u run.py $COMMON --pred_len 720 \
  --dropout 0.1 --learning_rate 0.0001 --lradj type1 \
  --train_epochs 10 --patience 3 \
  --no_geohcan \
  --itr 1 --des 'timesnet_base_720' --model_id tn_base_720 2>&1 | tee logs/tn_base_720.log

# ====================================================================
#  PART 2: GeoHCAN (基于 Exp D 最佳配置, 按 pred_len 调参)
#
#  通用: geohcan_d_model=512, lambda_geo=0.0, d_proj=16
#  
#  pred_len 越长:
#    - lambda_latent 适当增大 (锚定更强, 防长序列漂移)
#    - epochs/patience 增大 (长序列收敛慢)
#    - 长序列保留 use_proj_ln=1, detach_temp=1 (防止梯度爆炸)
# ====================================================================

GEO_COMMON="--geohcan_d_model 512 --d_proj 16 --lambda_geo 0.0"

echo ""
echo "============================================"
echo "  GeoHCAN pred_len=192 (lambda_latent=0.01)"
echo "============================================"
python -u run.py $COMMON --pred_len 192 \
  --dropout 0.1 --learning_rate 0.0001 --lradj type1 \
  --train_epochs 15 --patience 5 \
  $GEO_COMMON --lambda_latent 0.01 \
  --itr 1 --des 'timesnet_geo_192' --model_id tn_geo_192 2>&1 | tee logs/tn_geo_192.log

echo ""
echo "============================================"
echo "  GeoHCAN pred_len=336 (lambda_latent=0.02, 长序列保护ON)"
echo "============================================"
python -u run.py $COMMON --pred_len 336 \
  --dropout 0.1 --learning_rate 0.0001 --lradj type1 \
  --train_epochs 15 --patience 5 \
  $GEO_COMMON --lambda_latent 0.02 \
  --use_proj_ln 1 --detach_temp 1 \
  --itr 1 --des 'timesnet_geo_336' --model_id tn_geo_336 2>&1 | tee logs/tn_geo_336.log

echo ""
echo "============================================"
echo "  GeoHCAN pred_len=720 (lambda_latent=0.03, 长序列保护ON)"
echo "============================================"
python -u run.py $COMMON --pred_len 720 \
  --dropout 0.1 --learning_rate 0.0001 --lradj type1 \
  --train_epochs 15 --patience 5 \
  $GEO_COMMON --lambda_latent 0.03 \
  --use_proj_ln 1 --detach_temp 1 \
  --itr 1 --des 'timesnet_geo_720' --model_id tn_geo_720 2>&1 | tee logs/tn_geo_720.log

# ====================================================================
#  汇总结果
# ====================================================================
echo ""
echo "======================================================"
echo "  TimesNet ETTh1 全预测长度汇总"
echo "======================================================"
echo "原论文参考:  96→0.384  192→0.436  336→0.491  720→0.521"
echo "96 baseline: MSE=0.389"
echo "96 GeoHCAN:  MSE=0.390 (Exp D, lambda_latent=0.01)"
echo ""
echo "--- Baseline 192: ---"
grep "^mse:" logs/tn_base_192.log
echo "--- Baseline 336: ---"
grep "^mse:" logs/tn_base_336.log
echo "--- Baseline 720: ---"
grep "^mse:" logs/tn_base_720.log
echo ""
echo "--- GeoHCAN 192 (lat=0.01): ---"
grep "^mse:" logs/tn_geo_192.log
echo "--- GeoHCAN 336 (lat=0.02): ---"
grep "^mse:" logs/tn_geo_336.log
echo "--- GeoHCAN 720 (lat=0.03): ---"
grep "^mse:" logs/tn_geo_720.log
