#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-0}
#exp_dir="additional_loss_alpha0.4_lr3e-4"
exp_dir=${2:-"experiments/transformer_noise_ratio0.5_length_prediction_weight0.1_use_additional_loss_alpha1.0_lr3e-4"}
n_step=10
noise_ratio=0.5
length_margin=0.1
model_type="transformer"

for epoch in 160
do
python eval.py \
    --ckpt_path ./${exp_dir}/model/model_epoch${epoch}.pt \
    --save_dir ./${exp_dir}/evaluation \
    --dataset_path ./hubert_deberta_tar \
    --test_task eval_0 \
    --batch_size 128 \
    --gpu "0" \
    --n_step ${n_step} \
    --mask_token "[MASK]" \
    --noise_ratio ${noise_ratio} \
    --length_margin ${length_margin} \
    --model_type ${model_type}
done
