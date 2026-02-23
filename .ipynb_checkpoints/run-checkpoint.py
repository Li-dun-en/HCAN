import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

# =========================================================
#  固定随机种子 (确保复现性)
# =========================================================
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='SP-DEFI: Sorted-Projection Dynamic Evidential Fuzzy Inference')

# === Basic Config (基础配置) ===
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Informer',
                    help='model name, options: [Informer]')

# === Data Loader (数据加载) ===
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# === Forecasting Task (预测任务) ===
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# === Model Architecture (模型架构) ===
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# === Optimization (优化器) ===
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# =========================================================
# [SP-DEFI Specific Arguments] 核心参数
# =========================================================
parser.add_argument("--num_fine", default=4, type=int, help="Fine-grained classes (K_f)")
parser.add_argument("--num_coarse", default=2, type=int, help="Coarse-grained classes (K_c)")
parser.add_argument('--beta_scale', type=float, default=1.0, help='Scale factor for Soft-EDL gaussian width')

# 投影与几何参数
parser.add_argument('--d_proj', type=int, default=64, help='Projection dimension for RBF distance')
# rbf_gamma: Loss 使用固定值以保证监督目标稳定性，模型内部使用可学习 gamma
parser.add_argument('--rbf_gamma', type=float, default=1.0, help='RBF kernel width for Loss (固定值, 建议1.0)')

# 损失权重 (SP-DEFI+)
parser.add_argument('--lambda_direct', type=float, default=1.0, help='weight for main MSE loss')
parser.add_argument('--lambda_cls', type=float, default=0.1, help='KL分类损失权重 (SP-DEFI+)')
parser.add_argument('--lambda_reg', type=float, default=0.05, help='软回归损失权重 (SP-DEFI+)')
parser.add_argument('--lambda_acl', type=float, default=1.0, help='[legacy] unused')
parser.add_argument('--lambda_uac', type=float, default=1.0, help='[legacy] unused')

# 动态温度参数 (SP-DEFI+)
parser.add_argument('--alpha_temp', type=float, default=0.5, help='动态温度调节因子')

# =========================================================
# [Fix] GPU 参数修复
# =========================================================
# 将 type=bool 改为 type=int，避免 argparse 的布尔解析陷阱
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# [Critical Patch] 兼容 Informer.py 的参数读取
args.num_class = args.num_fine 

# GPU 逻辑检查
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # 建议修改 setting 生成逻辑
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_nf{}_proj{}_gamma{}_cls{}_reg{}_alp{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.num_fine,     
            args.d_proj,       
            args.rbf_gamma,
            args.lambda_cls,   # [新增]
            args.lambda_reg,   # [新增]
            args.alpha_temp,   # [新增]
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    # 建议修改 setting 生成逻辑
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_nf{}_proj{}_gamma{}_cls{}_reg{}_alp{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.num_fine,     
        args.d_proj,       
        args.rbf_gamma,
        args.lambda_cls,   # [新增]
        args.lambda_reg,   # [新增]
        args.alpha_temp,   # [新增]
        args.des, ii)

    exp = Exp(args)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
