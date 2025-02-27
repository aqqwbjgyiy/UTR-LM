#!/usr/bin/env python
# coding: utf-8
# 文件名: download_utr_data.py

import os
import requests
import gdown
import zipfile
import pandas as pd
from tqdm import tqdm
import shutil

def create_directory(directory):
    """创建目录，如果不存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def download_file(url, destination):
    """从URL下载文件到指定目标路径"""
    if os.path.exists(destination):
        print(f"文件已存在: {destination}")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    print(f"下载完成: {destination}")

def download_gdrive_file(file_id, destination):
    """从Google Drive下载文件"""
    if os.path.exists(destination):
        print(f"文件已存在: {destination}")
        return
    
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)
    print(f"下载完成: {destination}")

def download_gdrive_folder(folder_id, destination_dir):
    """从Google Drive下载整个文件夹"""
    create_directory(destination_dir)
    gdown.download_folder(id=folder_id, output=destination_dir, quiet=False)
    print(f"文件夹下载完成: {destination_dir}")

def main():
    # 创建主目录结构
    base_dir = "dataprepare"
    create_directory(base_dir)
    
    # 创建子目录
    pretrained_dir = os.path.join(base_dir, "Pretrained_Data")
    mrl_dir = os.path.join(base_dir, "MRL_Data")
    cellline_dir = os.path.join(base_dir, "CellLine_Data")
    model_dir = os.path.join(base_dir, "Model")
    
    create_directory(pretrained_dir)
    create_directory(mrl_dir)
    create_directory(cellline_dir)
    create_directory(model_dir)
    
    # 下载预训练数据
    print("\n=== 下载预训练数据 ===")
    pretrained_folder_id = "1_kmnYqYA5PNHQIxvwRgUn_RLZXS8Z7j3"
    download_gdrive_folder(pretrained_folder_id, pretrained_dir)
    
    # 下载MRL任务数据
    print("\n=== 下载MRL任务数据 ===")
    mrl_folder_id = "1csTXwy3LDCLKnzHHtcRsnu4LiJUEYHm3"
    download_gdrive_folder(mrl_folder_id, mrl_dir)
    
    # 下载TE和EL任务数据
    print("\n=== 下载TE和EL任务数据 ===")
    cellline_folder_id = "190oihtrwCxWjtDCK9kJzyhXPKxbr5xoR"
    download_gdrive_folder(cellline_folder_id, cellline_dir)
    
    # 下载预训练模型
    print("\n=== 下载预训练模型 ===")
    # 这里需要替换为实际的模型文件ID
    model_file_id = "YOUR_MODEL_FILE_ID"  # 需要替换为实际的文件ID
    model_file_path = os.path.join(model_dir, "ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl")
    
    print("\n注意: 预训练模型文件ID未提供，请手动下载或联系作者获取。")
    print("模型应保存到:", model_file_path)
    
    # 验证下载的数据
    print("\n=== 验证下载的数据 ===")
    
    # 检查预训练数据
    pretrained_files = os.listdir(pretrained_dir)
    print(f"预训练数据文件数量: {len(pretrained_files)}")
    if pretrained_files:
        print("预训练数据示例:", pretrained_files[:3])
    
    # 检查MRL数据
    mrl_files = os.listdir(mrl_dir)
    print(f"MRL数据文件数量: {len(mrl_files)}")
    if mrl_files:
        print("MRL数据示例:", mrl_files[:3])
        
        # 尝试读取一个MRL数据文件
        for file in mrl_files:
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(mrl_dir, file))
                    print(f"成功读取MRL数据文件: {file}")
                    print(f"数据形状: {df.shape}")
                    print(f"列名: {df.columns.tolist()}")
                    break
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {e}")
    
    # 检查细胞系数据
    cellline_files = os.listdir(cellline_dir)
    print(f"细胞系数据文件数量: {len(cellline_files)}")
    if cellline_files:
        print("细胞系数据示例:", cellline_files[:3])
    
    print("\n数据下载和验证完成！")
    print(f"所有数据已保存到 {os.path.abspath(base_dir)} 目录")

if __name__ == "__main__":
    main()
