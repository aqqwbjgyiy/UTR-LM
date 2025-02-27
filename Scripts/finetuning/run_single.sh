#!/bin/bash

# 设置分布式训练的环境变量
export WORLD_SIZE=1
export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
export RANK=0
export LOCAL_RANK=0

# 启动单卡训练
python main.py --device_ids 0 --local_rank 0