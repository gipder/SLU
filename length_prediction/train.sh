#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

final_epoch=100
num_samples=32
task="garbage"

python train.py \
    --save_dir ./${task} \
    --reset_save_dir True \
    --make_model_dir True \
    --dataset_path ./hubert_deberta_tar \
    --final_epoch ${final_epoch} \
    --train_task train \
    --eval_task eval \
    --batch_size 32 \
    --gpu "0" \
    --num_samples ${num_samples} \
    --mask_token "[MASK]" \
    --debugging False \
    --print_hist True
