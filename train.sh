#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate hif-vla

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b  \
  --data_root_dir xxx \
  --dataset_name libero_10_no_noops \
  --run_root_dir /xxx/hifvla/ckpt \
  --use_l1_regression True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "xxx" \
  --wandb_project "xxx" \
  --run_id_note libero_10--parallel_dec_decoder_2--l1_head--8_acts_chunk--continuous_acts--twoview--proprio_state \
  --history_length 8
