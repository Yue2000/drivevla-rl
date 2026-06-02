#!/usr/bin/env bash

NGPUS=1  # 快速测试用 1 卡
DATAPATH='/mnt/datasets/zhangzhang/navsim/download/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl'
MODEL_NAME_OR_PATH="/mnt/datasets/zyr/VLA_Emu/pretrained_models/Emu3-Stage1"
EXP_NAME=train_rl_stage1_offline_only

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=4

python train/train_rl_stage1.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --data_path ${DATAPATH} \
  --action_tokenizer_path /mnt/datasets/zyr/VLA_Emu/pretrained_models/fast \
  --actions_format fast \
  --output_dir logs/${EXP_NAME} \
  --per_device_train_batch_size 2 \
  --max_steps 100 \
  --learning_rate 5e-5 \
  --actor_lr 3e-4 \
  --critic_lr 3e-4 \
  --latent_lr 1e-4 \
  --weight_decay 0.1 \
  --max_grad_norm 5.0 \
  --save_steps 50 \
  --logging_steps 10 \
  --eval_strategy no \
  --gradient_checkpointing True \
  --seed 0