#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-2}
#exp_dir="additional_loss_alpha0.4_lr3e-4"
exp_dir=${2:-"experiments/baseline_depth8_lr1e-4"}
depth=8
model_type="transformer"

for epoch in 200 180 160 140 120 100 80 60
do
python eval.py \
    --ckpt_path ./${exp_dir}/model/model_epoch${epoch}.pt \
    --save_dir ./${exp_dir}/evaluation2 \
    --dataset_path ../data/slu/hubert_deberta_cache_tar \
    --test_task test \
    --batch_size 64 \
    --gpu "0" \
    --mask_token "[MASK]" \
    --model_type ${model_type} \
    --depth ${depth} \
    --norm_first True \
    --debugging False \
    --debugging_num 1024
done
