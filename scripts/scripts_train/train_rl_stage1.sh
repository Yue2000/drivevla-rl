#!/usr/bin/env bash

WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=23459
NGPUS=5

DATAPATH='/mnt/datasets/zhangzhang/navsim/download/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s_real_path.pkl'
ACTION_TOKENIZER_PATH="configs/fast"
MODEL_NAME_OR_PATH="/mnt/datasets/zyr/VLA_Emu/pretrained_models/Emu3-Stage1"
EXP_NAME=train_rl_stage1_offline_only

export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1,5,7,8,9

torchrun \
  --nproc_per_node=${NGPUS} \
  --nnodes=1 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train_rl_stage1.py \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --actions_format fast \
  --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
  --deepspeed scripts/sft/zero3_offload.json \
  --output_dir logs/${EXP_NAME} \
  --data_path ${DATAPATH} \
  --per_device_train_batch_size 2 \
  --max_steps 5000 \
  --learning_rate 5e-5 \
  --actor_lr 3e-4 \
  --critic_lr 3e-4 \
  --latent_lr 1e-4 \
  --weight_decay 0.1 \
  --max_grad_norm 5.0 \
  --bf16 True \
  --tf32 True \
  --save_steps 500 \
  --logging_steps 50 \
  --eval_strategy no \
  --gradient_checkpointing True \
  --seed 0