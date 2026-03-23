#!/bin/bash
#--asr_model_path ../finetune_asr/outputs/test_hubert_stop_lora_lr3e-4/epoch_030 \


export CUDA_VISIBLE_DEVICES=0
python ./extract_feature.py \
    --batch_size 8 \
    --cache_dir ../data/slu/hubert_deberta_cache_wt_asr_wer20 \
    --manifest_dir ../data/stop/manifests \
    --audio_prefix ~/work/DB/STOP/stop
