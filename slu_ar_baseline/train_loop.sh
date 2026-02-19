#!/bin/bash

final_epoch=200
valid_num_samples=512
model_type="transformer"
exp_dir="experiments"
lr=3e-4

# Array of depth values and corresponding GPUs
depths=(1 2 4 6 8)
gpus=(0 1 2 3 4)

# Create logs directory
mkdir -p logs

# Launch jobs in parallel
for i in {0..4}; do
    depth=${depths[$i]}
    gpu=${gpus[$i]}
    task="baseline_depth${depth}_lr${lr}"
    save_dir="${exp_dir}/${task}"
    log_file="logs/${task}.log"
    
    echo "Starting job: depth=${depth} on GPU ${gpu} (log: ${log_file})"
    
    CUDA_VISIBLE_DEVICES=${gpu} python train.py \
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
        --verbose False > ${log_file} 2>&1 &
done

# Wait for all background jobs to complete
echo "All 5 jobs launched. Monitoring progress..."
echo "To view logs in real-time, use: tail -f logs/baseline_depth*"

wait

echo "All jobs completed!"
