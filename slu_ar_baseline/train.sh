#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-"0"}

final_epoch=200
valid_num_samples=512
model_type="transformer"
exp_dir="experiments"
depth=6
lr=1e-4
task="baseline_depth${depth}_lr${lr}"
save_dir="${exp_dir}/${task}"
#task="additional_loss_alpha${alpha}_lr3e-4"

python train.py \
    --save_dir ${save_dir} \
    --reset_save_dir True \
    --make_model_dir True \
    --dataset_path ../data/slu/hubert_deberta_cache_tar \
    --final_epoch ${final_epoch} \
    --train_task train \
    --eval_task eval \
    --eval_epoch 5 \
    --batch_size 32 \
    --gpu "0" \
    --valid_num_samples ${valid_num_samples} \
    --model_type ${model_type} \
    --lr ${lr} \
    --depth ${depth} \
    --mask_token "[MASK]" \
    --debugging False \
    --verbose False 
