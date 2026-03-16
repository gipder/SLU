#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-0}
#exp_dir="additional_loss_alpha0.4_lr3e-4"

depth=6
model_type="transformer"
exp_dir="experiments/augmentation_span_mask_depth${depth}_lr3e-4"

for epoch in 200
do
python eval.py \
    --ckpt_path ./${exp_dir}/model/model_epoch${epoch}.pt \
    --save_dir ./${exp_dir}/evaluation \
    --dataset_path ../data/slu/hubert_deberta_cache_tar \
    --test_task test \
    --batch_size 128 \
    --gpu "0" \
    --mask_token "[MASK]" \
    --model_type ${model_type} \
    --depth ${depth} \
    --norm_first True \
    --debugging False \
    --debugging_num 1024
done
