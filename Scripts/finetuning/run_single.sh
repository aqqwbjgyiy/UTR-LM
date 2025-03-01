#!/bin/bash

# 设置分布式训练的环境变量
export WORLD_SIZE=1
export MASTER_ADDR='localhost'
# 修改端口号，避免端口冲突
export MASTER_PORT='23456'
export RANK=0
export LOCAL_RANK=0

# 设置多进程相关环境变量
export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore"

# 启动单卡训练
python main.py --device_ids 0 --local_rank 0 --layers 6 --num_workers 0