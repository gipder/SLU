#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-"5"}

final_epoch=300
valid_num_samples=512
n_step=5
alpha=1.0
use_additional_loss=True
length_loss_weight="0.1"
noise_ratio=0.5
model_type="transformer"
depth=2
exp_dir="experiments"
task="${model_type}_depth${depth}_noise_ratio${noise_ratio}_length_prediction_weight${length_loss_weight}_use_additional_loss_alpha${alpha}_lr3e-4"
save_dir="${exp_dir}/${task}"
#task="additional_loss_alpha${alpha}_lr3e-4"

python train.py \
    --save_dir ${save_dir} \
    --reset_save_dir False \
    --make_model_dir True \
    --dataset_path ./hubert_deberta_tar \
    --final_epoch ${final_epoch} \
    --train_task train \
    --eval_task eval \
    --eval_epoch 5 \
    --batch_size 32 \
    --gpu "0" \
    --n_step ${n_step} \
    --use_additional_loss ${use_additional_loss} \
    --alpha ${alpha} \
    --length_loss_weight ${length_loss_weight} \
    --valid_num_samples ${valid_num_samples} \
    --model_type ${model_type} \
    --depth ${depth} \
    --noise_ratio ${noise_ratio} \
    --mask_token "[MASK]" \
    --debugging False \
    --verbose False  
