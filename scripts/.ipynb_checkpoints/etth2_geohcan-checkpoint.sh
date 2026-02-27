#!/bin/bash
# =============================================================
# ETTh2 GeoHCAN 实验脚本
# 
# 数据集特点: ETTh2 波动较大, 需要针对性调参
# 调参策略:
#   1. 学习率: 偏低 (5e-5), 防止波动数据上过拟合
#   2. dropout: 短序列0.1(增强正则化), 长序列0.05(保留信息)
#   3. 短序列(96,192): energy_gate=1, proj_ln=0, detach_temp=0
#   4. 长序列(336,720): energy_gate=0, proj_ln=1, detach_temp=1
#   5. lambda_latent: 短序列0.05, 长序列适当降低防止过度约束
#   6. train_epochs=12, patience=4 给予充分收敛时间
#   7. factor=3 (ProbSparse, ETT系列验证最优)
# =============================================================

# 公共参数
MODEL=Informer
DATA=ETTh2
ROOT_PATH=./dataset/ETT-small/
DATA_PATH=ETTh2.csv
FEATURES=M
SEQ_LEN=96
LABEL_LEN=48
ENC_IN=7
DEC_IN=7
C_OUT=7
D_MODEL=512
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=2048
FACTOR=3
NUM_FINE=4
D_PROJ=64
BATCH_SIZE=32
ITR=1
TRAIN_EPOCHS=12
PATIENCE=4
LAMBDA_CLS=0.1
LAMBDA_REG=0.05
LAMBDA_GEO=0.0
ALPHA_TEMP=0.5
GATE_POWER=2

echo "============================================"
echo " ETTh2 GeoHCAN 全长度实验"
echo " pred_len: 96 / 192 / 336 / 720"
echo "============================================"

# =============================================================
# pred_len = 96 (短序列)
# 策略: energy_gate=1, proj_ln=0, detach_temp=0
#        lr=5e-5, dropout=0.1, lambda_latent=0.05
# =============================================================
echo ""
echo "============================================"
echo " [1/4] ETTh2 pred_len=96"
echo "   lr=5e-5, dropout=0.1, energy_gate=ON"
echo "============================================"
python -u run.py \
  --is_training 1 \
  --model $MODEL \
  --model_id ETTh2_96_96 \
  --data $DATA \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 96 \
  --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
  --d_model $D_MODEL --n_heads $N_HEADS \
  --e_layers $E_LAYERS --d_layers $D_LAYERS --d_ff $D_FF \
  --factor $FACTOR \
  --dropout 0.1 \
  --num_fine $NUM_FINE --d_proj $D_PROJ \
  --lambda_cls $LAMBDA_CLS --lambda_reg $LAMBDA_REG \
  --lambda_geo $LAMBDA_GEO --lambda_latent 0.05 \
  --alpha_temp $ALPHA_TEMP \
  --use_proj_ln 0 --detach_temp 0 --use_energy_gate 1 \
  --gate_power $GATE_POWER \
  --learning_rate 0.00005 \
  --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
  --patience $PATIENCE --itr $ITR \
  --des 'ETTh2_GeoHCAN' 2>&1 | tee logs/ETTh2/etth2_96.log

echo ""
echo ">>> pred_len=96 完成"
echo ""

# =============================================================
# pred_len = 192 (中短序列)
# 策略: energy_gate=1, proj_ln=0, detach_temp=0
#        lr=5e-5, dropout=0.1, lambda_latent=0.05
# =============================================================
echo "============================================"
echo " [2/4] ETTh2 pred_len=192"
echo "   lr=5e-5, dropout=0.1, energy_gate=ON"
echo "============================================"
python -u run.py \
  --is_training 1 \
  --model $MODEL \
  --model_id ETTh2_96_192 \
  --data $DATA \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 192 \
  --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
  --d_model $D_MODEL --n_heads $N_HEADS \
  --e_layers $E_LAYERS --d_layers $D_LAYERS --d_ff $D_FF \
  --factor $FACTOR \
  --dropout 0.1 \
  --num_fine $NUM_FINE --d_proj $D_PROJ \
  --lambda_cls $LAMBDA_CLS --lambda_reg $LAMBDA_REG \
  --lambda_geo $LAMBDA_GEO --lambda_latent 0.05 \
  --alpha_temp $ALPHA_TEMP \
  --use_proj_ln 0 --detach_temp 0 --use_energy_gate 1 \
  --gate_power $GATE_POWER \
  --learning_rate 0.00005 \
  --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
  --patience $PATIENCE --itr $ITR \
  --des 'ETTh2_GeoHCAN' 2>&1 | tee logs/ETTh2/etth2_192.log

echo ""
echo ">>> pred_len=192 完成"
echo ""

# =============================================================
# pred_len = 336 (长序列)
# 策略: energy_gate=0, proj_ln=1, detach_temp=1 (长序列保护)
#        lr=5e-5, dropout=0.05, lambda_latent=0.02
# =============================================================
echo "============================================"
echo " [3/4] ETTh2 pred_len=336"
echo "   lr=5e-5, dropout=0.05, 长序列保护ON"
echo "============================================"
python -u run.py \
  --is_training 1 \
  --model $MODEL \
  --model_id ETTh2_96_336 \
  --data $DATA \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 336 \
  --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
  --d_model $D_MODEL --n_heads $N_HEADS \
  --e_layers $E_LAYERS --d_layers $D_LAYERS --d_ff $D_FF \
  --factor $FACTOR \
  --dropout 0.05 \
  --num_fine $NUM_FINE --d_proj $D_PROJ \
  --lambda_cls 0.05 --lambda_reg 0.02 \
  --lambda_geo $LAMBDA_GEO --lambda_latent 0.02 \
  --alpha_temp $ALPHA_TEMP \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --gate_power $GATE_POWER \
  --learning_rate 0.00005 \
  --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
  --patience $PATIENCE --itr $ITR \
  --des 'ETTh2_GeoHCAN' 2>&1 | tee logs/ETTh2/etth2_336.log

echo ""
echo ">>> pred_len=336 完成"
echo ""

# =============================================================
# pred_len = 720 (超长序列)
# 策略: energy_gate=0, proj_ln=1, detach_temp=1 (长序列保护)
#        lr=1e-4 (超长序列需稍高LR加速收敛), dropout=0.05
#        lambda_latent=0.01, gate_power=3 (更保守注入)
# =============================================================
echo "============================================"
echo " [4/4] ETTh2 pred_len=720"
echo "   lr=1e-4, dropout=0.05, 长序列保护ON"
echo "============================================"
python -u run.py \
  --is_training 1 \
  --model $MODEL \
  --model_id ETTh2_96_720 \
  --data $DATA \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 720 \
  --enc_in $ENC_IN --dec_in $DEC_IN --c_out $C_OUT \
  --d_model $D_MODEL --n_heads $N_HEADS \
  --e_layers $E_LAYERS --d_layers $D_LAYERS --d_ff $D_FF \
  --factor $FACTOR \
  --dropout 0.05 \
  --num_fine $NUM_FINE --d_proj $D_PROJ \
  --lambda_cls 0.05 --lambda_reg 0.02 \
  --lambda_geo $LAMBDA_GEO --lambda_latent 0.01 \
  --alpha_temp $ALPHA_TEMP \
  --use_proj_ln 1 --detach_temp 1 --use_energy_gate 0 \
  --gate_power 3 \
  --learning_rate 0.0001 \
  --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
  --patience $PATIENCE --itr $ITR \
  --des 'ETTh2_GeoHCAN' 2>&1 | tee logs/ETTh2/etth2_720.log

echo ""
echo "============================================"
echo " ETTh2 全部实验完成!"
echo "============================================"
echo ""
echo "结果汇总:"
echo "  pred_len=96:  logs/ETTh2/etth2_96.log"
echo "  pred_len=192: logs/ETTh2/etth2_192.log"
echo "  pred_len=336: logs/ETTh2/etth2_336.log"
echo "  pred_len=720: logs/ETTh2/etth2_720.log"
echo ""
echo "可视化结果保存在 test_results/ 目录下"
