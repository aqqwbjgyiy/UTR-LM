#!/usr/bin/env python
# coding: utf-8

import os
from config import config
import argparse
from train import train_model

def parse_args():
    parser = argparse.ArgumentParser()
    # 训练设备相关参数
    parser.add_argument('--device_ids', type=str, default='0,1,2', help="Training Devices")
    parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
    parser.add_argument('--log_interval', type=int, default=50, help="Log Interval")
    parser.add_argument('--seed', type=int, default=1337)

    # 模型训练相关参数
    parser.add_argument('--prefix', type=str, default='ESM2SISS_FS4.1.ep93.1e-2.dr5')
    parser.add_argument('--label_type', type=str, default='rl')
    parser.add_argument('--seq_type', type=str, default='utr')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--cnn_layers', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=40)
    parser.add_argument('--dropout3', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--patience', type=int, default=0)
    
    # 训练数据相关参数
    parser.add_argument('--train_file', type=str, default='4.1_train_data_GSM3130435_egfp_unmod_1.csv')
    # 修改模型文件路径参数
    parser.add_argument('--modelfile', type=str, default=os.path.join(config.model_dir, 'ESM2_1.4_five_species_TrainLossMin_6layers_16heads_128embedsize_4096batchToks.pkl'))
    parser.add_argument('--finetune_modelfile', type=str, default=os.path.join(config.model_dir, 'CVESM2lr1e-5_DDP9.1_unmod_1_10folds_rl_LabelScalerTrue_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.2_finetuneTrue_huberlossTrue_magicFalse_fold0_epoch19_lr0.1.pt'))

    # 功能开关参数
    parser.add_argument('--test1fold', action='store_true')
    parser.add_argument('--huber_loss', action='store_true')
    parser.add_argument('--load_wholemodel', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--scaler', action='store_true')
    parser.add_argument('--log2', action='store_true')
    parser.add_argument('--avg_emb', action='store_true')
    parser.add_argument('--bos_emb', action='store_true')
    parser.add_argument('--train_atg', action='store_true')
    parser.add_argument('--train_n_atg', action='store_true')
    parser.add_argument('--magic', action='store_true')
    
    parser.add_argument('--init_epochs', type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train_model(args)

if __name__ == "__main__":
    main()