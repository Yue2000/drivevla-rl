#!/usr/bin/env bash

NGPUS=8
DATAPATH='/mnt/datasets/zhangzhang/navsim/download/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
MODEL_NAME_OR_PATH="logs/train_rl_stage1_offline_only/final"  # 使用阶段 1 的检查点
EXP_NAME=train_rl_stage2_with_dynamics

export PYTHONPATH=$(pwd)

torchrun \
  --nproc_per_node=${NGPUS} \
  train/train_rl_stage2.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --output_dir logs/${EXP_NAME} \
  --data_path ${DATAPATH} \
  --per_device_train_batch_size 8 \
  --max_steps 5000 \
  --actor_lr 3e-4 \
  --critic_lr 3e-4 \
  --latent_lr 1e-4 \
  --dynamics_lr 1e-4 \
  --weight_decay 0.1 \
  --save_steps 500 \
  --logging_steps 50 \
  --bf16 True \
  --seed 0