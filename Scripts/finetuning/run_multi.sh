#!/bin/bash

# 设置GPU数量
NUM_GPUS=3

# 设置分布式训练的环境变量
export WORLD_SIZE=$NUM_GPUS
export MASTER_ADDR='localhost'
export MASTER_PORT='12355'

# 启动多卡训练
for ((i=0; i<$WORLD_SIZE; i++))
do
    export RANK=$i
    export LOCAL_RANK=$i
    python main.py --device_ids 0,1,2 --local_rank $i &
done

wait