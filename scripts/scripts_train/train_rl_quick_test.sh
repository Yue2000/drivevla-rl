#!/usr/bin/env bash

# 快速测试：1 卡，100 步
DATAPATH='/mnt/datasets/zhangzhang/navsim/download/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
MODEL_NAME_OR_PATH='/mnt/datasets/yc/VLA_Emu/Emu3_Flow_Matching_Action_Expert_PDMS_87.2'

export PYTHONPATH=$(pwd)

python train/train_rl_stage1.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --data_path ${DATAPATH} \
  --output_dir logs/test_stage1 \
  --per_device_train_batch_size 2 \
  --max_steps 100 \
  --actor_lr 3e-4 \
  --critic_lr 3e-4 \
  --latent_lr 1e-4 \
  --save_steps 50 \
  --logging_steps 10 \
  --eval_strategy no \
  --seed 0