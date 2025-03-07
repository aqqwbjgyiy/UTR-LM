import os
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
from esm import FastaBatchedDataset
from torch.utils.data.distributed import DistributedSampler
import torch
from config import config

def load_data(args, seed):
    """加载并预处理数据"""
    # 检查数据文件是否存在
    original_train_file = args.train_file.replace('_delATG', '').replace('_ATGtoUNK', '').replace('_ATGtoCAA', '')
    train_file_path = os.path.join(config.data_dir, original_train_file)
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Training file not found: {train_file_path}\n"
                              f"Please ensure the data directory structure is correct:\n"
                              f"- Data directory: {config.data_dir}\n"
                              f"- Required file: {original_train_file}")

    original_train_data = pd.read_csv(train_file_path)

    # 检查训练文件
    train_file_path = os.path.join(config.data_dir, args.train_file)
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Training file not found: {train_file_path}")
    train_data = pd.read_csv(train_file_path)
    
    if args.train_atg:
        data = train_data[train_data[args.seq_type].str.contains('ATG')]
    elif args.train_n_atg:
        data = train_data[~train_data[args.seq_type].str.contains('ATG')]
    else:
        data = deepcopy(train_data)
    
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    if args.log2:
        data = data[data[args.label_type] != 0]
        data[f'{args.label_type}_log2'] = data[args.label_type].apply(np.log2)

    # 检查测试文件
    test_file = args.train_file.replace('train', 'test')
    test_file_path = os.path.join(config.data_dir, test_file)
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
    e_test = pd.read_csv(test_file_path)
    
    e_test = e_test.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    if args.log2:
        e_test = e_test[e_test[args.label_type] != 0]
        e_test[f'{args.label_type}_log2'] = e_test[args.label_type].apply(np.log2)
        
    return original_train_data, data, e_test

def generate_dataset_dataloader(e_data, obj_col, batch_toks, alphabet):
    """生成数据加载器"""
    dataset = FastaBatchedDataset(e_data.loc[:,obj_col], e_data.utr, mask_prob=0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    dataloader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=alphabet.get_batch_converter(),
                                         batch_sampler=batches,
                                         shuffle=False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader

def generate_trainbatch_loader(e_data, obj_col, batch_toks, alphabet, num_workers=8):
    """生成训练批次加载器"""
    dataset = FastaBatchedDataset(e_data.loc[:,obj_col], e_data.utr, mask_prob=0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)
    batches_sampler = DistributedSampler(batches, shuffle=True)
    batches_loader = torch.utils.data.DataLoader(batches,
                                              batch_size=1,
                                              num_workers=num_workers,
                                              sampler=batches_sampler)
    print(f"{len(dataset)} sequences")
    print(f'{len(batches)} batches')
    return dataset, batches, batches_sampler, batches_loader