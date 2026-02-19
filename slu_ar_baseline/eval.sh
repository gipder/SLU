#!/bin/bash

export CUDA_VISIBLE_DEVICES=${2:-0}
#exp_dir="additional_loss_alpha0.4_lr3e-4"
exp_dir=${1:-"experiments/baseline_depth2_lr3e-4"}
depth=2
model_type="transformer"

for epoch in 180
do
python eval.py \
    --ckpt_path ./${exp_dir}/model/model_epoch${epoch}.pt \
    --save_dir ./${exp_dir}/evaluation \
    --dataset_path ../data/slu/hubert_deberta_cache_tar \
    --test_task test \
    --batch_size 64 \
    --gpu "0" \
    --mask_token "[MASK]" \
    --model_type ${model_type} \
    --depth ${depth}
done
