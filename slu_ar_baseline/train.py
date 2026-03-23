import os
import sys
import argparse
from argparse import ArgumentParser
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
import json
from dataclasses import asdict
import logging
import time

import glob
from transformers import AutoProcessor, HubertForCTC
from torchvision import models, transforms
from torchvision.datasets import MNIST
from transformers import HubertModel
from torch.nn.utils.rnn import pad_sequence
#from speech_featured_unet import DiscreteContextUnet
from torch.amp import autocast, GradScaler

from model import ARModel, ARModelConfig
from hubert_deberta_dataset import HuBERTandDeBERTaDataset
from predict_model import SelfPromptARModel, SelfPromptARModelConfig
from hubert_deberta_dataset import hubert_and_deberta_dataset_collate_fn
from hubert_deberta_dataset import BatchSampler
#from sampling import sampling_batch, sampling_debugging
#from custom_path import UniformDiscreteProbPath

from jiwer import wer
from utils import str2bool, remove_module_prefix
from augmentation import build_augmentor
from utils import setup_logger, class_name


def build_parser():
    p = argparse.ArgumentParser(description="Train SLU with HuBERT + DeBERTa features")

    # ---- training ----
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--total_step", type=int, default=800000)
    p.add_argument("--final_epoch", type=int, default=100,)
    p.add_argument("--log_step", type=int, default=500, help="Logging step interval")
    p.add_argument("--eval_step", type=int, default=5000, help="Evaluation step interval")
    p.add_argument("--eval_epoch", type=int, default=5, help="Evaluation epoch interval")
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate by warmup step")
    p.add_argument("--warmup_step", type=int, default=2500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default=None, help="If None, auto-generated from lr.")
    p.add_argument("--make_model_dir", type=str2bool, default=True, help="Whether to make model save_dir")
    p.add_argument("--reset_save_dir", type=str2bool, default=False, help="Whether to reset save_dir if exists")
    p.add_argument("--save_step", type=int, default=50000, help="Not using currently")
    #p.add_argument("--uniform", type=str2bool, default=False)
    #p.add_argument("--loss_type", type=str,
    #               default="ce", choices=["ce", "gkl"], help="ce or gkl(generalized KL)")
    #p.add_argument("--use_additional_loss_only", action="store_true", default=False,
    #               help="For comparison, use only additional loss term")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_path", type=str, default=None, help="Path to load checkpoint")
    p.add_argument("--gpu", type=str, default="0", help="GPU ids separated by comma, e.g., '0,1,2'")
    p.add_argument("--use_tar", type=str2bool, default=True, help="Whether to use tarred dataset")
    # additional loss
    #p.add_argument("--alpha", type=float, default=0.1, help="Weight for additional loss term")
    #p.add_argument("--use_additional_loss", type=str2bool, default=False,
    #               help="Whether to use additional loss term")

    # ---- model dims / arch ----
    ## for DFM model
    p.add_argument("--vocab_size", type=int, default=47)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--audio_dim", type=int, default=1024)
    p.add_argument("--text_dim", type=int, default=1024)
    p.add_argument("--max_output_length", type=int, default=512, help="Maximum output length during inference")
    #p.add_argument("--noise_ratio", type=float, default=0.5, help="Noise ratio for UniformDiscreteProbPath")
    #p.add_argument("--n_step", type=int, default=5, help="Number of sampling steps during inference")
    p.add_argument("--model_type", type=str, choices=["transformer", "encoder_decoder_transformer", "fused_transformer"], default="transformer")
    p.add_argument("--norm_first", type=str2bool, default=True, help="Whether to apply layer normalization before attention and FFN")
    p.add_argument("--condition_type", type=str, choices=["both", "audio", "text"], default="both", help="Type of conditioning for the model")
    ## for length predictor
    #p.add_argument("--embed_dim", type=int, default=1024)
    #p.add_argument("--length_hidden_dim", type=int, default=512)
    ##p.add_argument("--max_target_positions", type=int, default=256)
    #p.add_argument("--length_dropout", type=float, default=0.1)
    #p.add_argument("--length_condition", type=str, choices=["audio", "text", "both"], default="text")
    #p.add_argument("--length_margin", type=float, default=0.1)
    #p.add_argument("--length_loss_weight", type=float, default=1.0)

    # ---- data / tokenization ----,
    p.add_argument("--dataset_path", type=str, default="./hubert_deberta_cache_retrial")
    p.add_argument("--train_task", type=str, default="train")
    p.add_argument("--eval_task", type=str, default="eval")
    p.add_argument("--test_task", type=str, default="test")
    p.add_argument("--tokenizer_model_name", type=str, default="facebook/hubert-large-ls960-ft")
    p.add_argument("--mask_token", type=str, default="[MASK]")
    p.add_argument("--use_intent_token", type=str2bool, default=False,
                   help="Whether to add INTENT-derived tokens (and sentencepiece variants) to tokenizer")
    p.add_argument("--valid_num_samples", type=int, default=2048, help="Number of samples to use for validation")
    #p.add_argument("--shuffle_train", type=bool, default=True)

    # ---- debugging ----
    # debugging for dataset
    p.add_argument("--dataset_debugging", type=str2bool, default=False, help="Enable debugging mode for dataset")
    p.add_argument("--dataset_debugging_num", type=int, default=128, help="How many samples are used in debugging dataset")
    # debugging for sampling and others...
    p.add_argument("--debugging", type=str2bool, default=False, help="Enable debugging mode in train_dfm()")
    p.add_argument("--verbose", type=str2bool, default=False, help="Enable verbose logging during sampling")

    # ---- device ----
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    # ---- data augmentation ----
    p.add_argument("--use_augment", type=str2bool, default=False, help="Enable feature augmentation")
    p.add_argument("--augment_type", type=str, default="span_mask",
                   help="Augmentation type(s). 단일: 'span_mask' / 조합: 'span_mask,gaussian_noise'")
    p.add_argument("--augment_audio_noise_std", type=float, default=0.02,
                   help="Std for Gaussian noise on audio features (HuBERT-large 권장: 0.01~0.05)")
    p.add_argument("--augment_text_noise_std", type=float, default=0.01,
                   help="Std for Gaussian noise on text features (DeBERTa 권장: 0.005~0.02)")
    p.add_argument("--augment_noise_schedule", type=str, default="constant",
                   choices=["constant", "linear_increase", "linear_decrease"],
                   help="Schedule for noise std")
    p.add_argument("--augment_noise_schedule_steps", type=int, default=0,
                   help="Steps for noise schedule (0 = constant)")
    p.add_argument("--augment_use_layernorm", type=str2bool, default=False,
                   help="Apply LayerNorm before noise")
    p.add_argument("--augment_audio_mask_ratio", type=float, default=0.1,
                   help="Audio span mask ratio (0~1)")
    p.add_argument("--augment_text_mask_ratio", type=float, default=0.1,
                   help="Text span mask ratio (0~1)")
    p.add_argument("--augment_audio_mask_span", type=int, default=10,
                   help="Audio span length")
    p.add_argument("--augment_text_mask_span", type=int, default=3,
                   help="Text span length")

    # ---- self prompt ----
    p.add_argument("--use_self_prompt", type=str2bool, default=False,
                   help="Use SelfPromptARModel with intent prediction auxiliary loss")
    p.add_argument("--num_intent", type=int, default=80,
                   help="Number of intent classes for prompt predictor")
    p.add_argument("--intent_loss_weight", type=float, default=1.0,
                   help="Weight for predict_intent_loss in total loss")
    p.add_argument("--length_hidden_dim", type=int, default=128,
                   help="Hidden dim for MaskedIntentPredictionModule")
    p.add_argument("--length_dropout", type=float, default=0.1,
                   help="Dropout for MaskedIntentPredictionModule")
    p.add_argument("--prompt_tag_path", type=str, default="data/slu/INTENT",
                   help="Path to INTENT/TAG label file used for prompt table")

    return p


def apply_condition_type(args, audio_feats, audio_mask, text_feats, text_mask):
    if args.condition_type == "both":
        return audio_feats, audio_mask, text_feats, text_mask
    if args.condition_type == "audio":
        return audio_feats, audio_mask, None, None
    if args.condition_type == "text":
        return None, None, text_feats, text_mask
    raise ValueError(f"Invalid condition_type: {args.condition_type}")


def _get_base_model(model):
    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    return m


def _build_tag_token_patterns(tokenizer, prompt_table: List[str]) -> List[Tuple[int, List[int]]]:
    """TAG 문자열 테이블을 tokenizer id 시퀀스로 변환: [(tag_idx, [id...]), ...]"""
    patterns = []
    for tag_idx, tag in enumerate(prompt_table):
        ids = tokenizer.encode(tag, add_special_tokens=False)
        if len(ids) > 0:
            patterns.append((tag_idx, ids))
    return patterns


def _find_first_matching_tag_id(token_ids: List[int], tag_patterns: List[Tuple[int, List[int]]]) -> int:
    """토큰 시퀀스 내에서 가장 먼저 등장한 TAG 패턴의 tag_idx를 반환. 없으면 -1."""
    best_tag_id = -1
    best_pos = None
    for tag_idx, pattern in tag_patterns:
        n = len(pattern)
        if n == 0 or n > len(token_ids):
            continue
        for i in range(len(token_ids) - n + 1):
            if token_ids[i:i+n] == pattern:
                if best_pos is None or i < best_pos:
                    best_pos = i
                    best_tag_id = tag_idx
                break
    return best_tag_id


def _build_intent_gt_from_slu_ids(
    slus: torch.Tensor,
    slu_mask: Optional[torch.Tensor],
    tag_patterns: List[Tuple[int, List[int]]],
) -> torch.Tensor:
    """SLU target token ids에서 TAG 매칭 기반 intent GT id (B,)를 생성. 매칭 실패는 -1."""
    B, T = slus.shape
    out = torch.full((B,), -1, dtype=torch.long, device=slus.device)
    if not tag_patterns:
        return out

    for b in range(B):
        valid_len = int(slu_mask[b].sum().item()) if slu_mask is not None else T
        token_ids = slus[b, :valid_len].tolist()
        out[b] = _find_first_matching_tag_id(token_ids, tag_patterns)
    return out


def _build_prompt_prefix_lengths(
    intent_gt: Optional[torch.Tensor],
    prompt_table: List[str],
    tokenizer,
) -> Optional[torch.Tensor]:
    """intent GT 기준 정답 prompt prefix 길이(B,)를 생성. invalid GT는 0."""
    if intent_gt is None:
        return None

    lengths = torch.zeros_like(intent_gt, dtype=torch.long)
    for b in range(intent_gt.size(0)):
        idx = int(intent_gt[b].item())
        if 0 <= idx < len(prompt_table):
            lengths[b] = len(tokenizer.encode(prompt_table[idx], add_special_tokens=False))
    return lengths


def _resolve_label_path(label_path: str) -> Path:
    path = Path(label_path)
    if path.is_absolute():
        return path

    module_dir = Path(__file__).resolve().parent
    workspace_root = module_dir.parent
    candidates = [
        Path.cwd() / path,
        module_dir / path,
        workspace_root / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _load_intent_labels(label_path: str) -> List[str]:
    path = _resolve_label_path(label_path)
    if not path.exists():
        logger.warning(f"Intent label file not found: {path}")
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info(f"* Loaded {len(lines)} intent labels from: {path}")
    return lines


def _build_intent_token_candidates(intent_labels: List[str]) -> List[str]:
    """Build tokenizer tokens from full INTENT labels only."""
    spm_prefix = "▁"
    candidates = []

    for label in intent_labels:
        # INTENT appears at the beginning, so keep only sentencepiece-style boundary form.
        candidates.append(spm_prefix + label)

    # Stable deduplication while preserving insertion order.
    return list(dict.fromkeys(candidates))


# -------------------------
# Best Model Link Update
# -------------------------
def update_best_model_link(metric_value, best_metric_value, metric_name, is_better_fn, save_dir, ckpt_path, args):
    """
    Best model symlink를 업데이트합니다.

    Args:
        metric_value: 현재 metric 값
        best_metric_value: 지금까지의 최고 metric 값
        metric_name: metric 이름 (e.g., "valid_loss", "em")
        is_better_fn: 더 좋은지 판단하는 함수 (e.g., lambda x, y: x < y for loss, x > y for em)
        save_dir: 저장 디렉토리
        ckpt_path: 체크포인트 경로
        args: 학습 인자

    Returns:
        bool: best model이 업데이트되었는지 여부
    """
    if is_better_fn(metric_value, best_metric_value):
        if args.make_model_dir:
            best_link = os.path.join(save_dir, "model", f"best_{metric_name}.pt")
        else:
            best_link = os.path.join(save_dir, f"best_{metric_name}.pt")

        if os.path.islink(best_link) or os.path.exists(best_link):
            os.remove(best_link)

        os.symlink(os.path.basename(ckpt_path), best_link)
        base_path_best_link = os.path.basename(best_link)
        base_path_ckpt_path = os.path.basename(ckpt_path)

        logger.info(
            f"New best {metric_name}: {metric_value:.6f} "
            f"-> symlink: {base_path_best_link} "
            f"-> {base_path_ckpt_path}"
        )
        return True
    else:
        logger.info(
            f"{metric_name} did not improve: "
            f"{metric_value:.6f} (best: {best_metric_value:.6f})"
        )
        return False


# -------------------------
# Validation function
# -------------------------
# Validation function
# -------------------------
def validate_model(
    model,
    eval_dataset,
    tokenizer,
    args,
    sos_id,
    eos_id,
    device,
    valid_num_samples=2048,
    epoch=None,
):
    """
    Validation 수행 및 메트릭 계산
    매번 다른 샘플을 무작위로 선택하여 평가

    Args:
        dfm_model: 평가할 모델
        eval_dataset: 전체 evaluation dataset
        tokenizer: 토크나이저
        args: 학습 인자
        sos_id: 시작 토큰 ID
        eos_id: 종료 토큰 ID
        device: 학습 device
        valid_num_samples: validation에 사용할 샘플 개수 (기본값: 2048)
        epoch: 현재 epoch

    Returns:
        dict: 'dfm_wer', 'asr_wer', 'gt_wer', 'accuracy' 포함
    """
    model.eval()

    # 전체 dataset에서 무작위로 num_samples만큼 선택
    total_samples = len(eval_dataset)
    valid_num_samples_to_use = min(valid_num_samples, total_samples)

    if total_samples > valid_num_samples_to_use:
        random_indices = torch.randperm(total_samples)[:valid_num_samples_to_use]
        val_dataset = Subset(eval_dataset, random_indices)
        logger.info(f"* Validation: Sampled {valid_num_samples_to_use:,} from {total_samples:,} samples")
    else:
        val_dataset = eval_dataset
        logger.info(f"* Validation: Using all {total_samples:,} samples")

    # Validation용 DataLoader 생성
    val_sampler = BatchSampler(val_dataset,
                               batch_size=args.batch_size,
                               shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=hubert_and_deberta_dataset_collate_fn,
    )
    total_data_samples = len(val_dataset)

    # intent 정확도 측정 준비 (use_self_prompt 활성 시)
    tag_patterns = []
    prompt_table = []
    if getattr(args, 'use_self_prompt', False):
        m = _get_base_model(model)
        prompt_table = getattr(m, 'prompt_table', [])
        tag_patterns = _build_tag_token_patterns(tokenizer, prompt_table)

    with torch.no_grad():
        hyp_ids = []
        target_ids = []
        str_asr_hyps = []
        str_slus = []
        pred_intent_ids = []
        gt_intent_ids = []
        step = 0
        count = 0
        for batch in val_loader:
            audio_feats = batch["feat"]
            audio_feat_mask = batch["feat_mask"]
            text_feats = batch["text_feat"]
            text_feat_mask = batch["text_mask"]
            slus = batch["slu"]
            slu_mask = batch["slu_mask"]
            str_asr_hypothesis = batch["str_hyp"]
            str_slu_targets = batch["str_slu"]

            audio_feats = audio_feats.to(device) # B, T, D
            audio_feat_mask = audio_feat_mask.to(device)
            text_feats = text_feats.to(device)
            text_feat_mask = text_feat_mask.to(device)
            slus = slus.to(device)
            slu_mask = slu_mask.to(device)

            audio_feats, audio_feat_mask, text_feats, text_feat_mask = apply_condition_type(
                args,
                audio_feats,
                audio_feat_mask,
                text_feats,
                text_feat_mask,
            )

            slus = slus.to(device)
            slu_mask = slu_mask.to(device)

            batch_top_intent_ids = None
            use_retrieval_decode = getattr(args, 'use_self_prompt', False)
            m = _get_base_model(model)

            if use_retrieval_decode and hasattr(m, 'retrieve_prompts') and hasattr(m, 'decode_with_prompt'):
                # 1. 한 번만 retrieve
                retrieval = m.retrieve_prompts(
                    audio_feats=audio_feats,
                    text_feats=text_feats,
                    audio_mask=audio_feat_mask,
                    text_mask=text_feat_mask,
                    top_k=1,
                )
                batch_top_intent_ids = retrieval['top_intent_ids']
                #print(f"{retrieval['retrieved_prompts']=}")
                # 2. top-1 intent prefix tokenize
                prefix_id_list = []
                for row_labels in retrieval["retrieved_prompts"]:
                    ids = tokenizer.encode(row_labels[0])
                    prefix_id_list.append(ids)
                max_p = max(len(ids) for ids in prefix_id_list)
                pad_id = getattr(tokenizer, "pad_token_id", 0)
                padded = [ids + [pad_id] * (max_p - len(ids)) for ids in prefix_id_list]
                prompt_prefix_ids = torch.tensor(padded, dtype=torch.long)

                # 3. prompt-conditioned decode (retrieve는 위에서 완료)
                generated = m.decode_with_prompt(
                    audio_feats=audio_feats,
                    text_feats=text_feats,
                    audio_mask=audio_feat_mask,
                    text_mask=text_feat_mask,
                    prompt_prefix_ids=prompt_prefix_ids,
                    max_output_length=args.max_output_length,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    device=device,
                )
            else:
                generated = model.decode(
                    audio_feats=audio_feats,
                    text_feats=text_feats,
                    audio_mask=audio_feat_mask,
                    text_mask=text_feat_mask,
                    max_output_length=args.max_output_length,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    do_sample=False,
                    device=device,
                )

            hyp_ids.extend(generated.cpu().tolist())
            target_ids.extend(slus.cpu().tolist())
            str_asr_hyps.extend(str_asr_hypothesis)
            str_slus.extend(str_slu_targets)

            if use_retrieval_decode and tag_patterns and batch_top_intent_ids is not None:
                pred_intent_ids.extend(batch_top_intent_ids[:, 0].detach().cpu().tolist())
                intent_gt_batch = _build_intent_gt_from_slu_ids(slus, slu_mask, tag_patterns)
                gt_intent_ids.extend(intent_gt_batch.cpu().tolist())

            step += 1
            count += slus.size(0)

            if step % 10 == 0:
                logger.info(f"Evaluation step {step:,}/{len(val_loader):,} completed.")
                logger.info(f"  Processed {count:,}/{total_data_samples:,} samples.")

    # Decode predictions
    str_hyps = []
    str_targets = []
    correct_predictions = 0

    assert valid_num_samples_to_use == len(hyp_ids)
    for b in range(valid_num_samples_to_use):
        hyp_id = hyp_ids[b]
        target_id = target_ids[b]
        str_asr_hyp = str_asr_hyps[b]
        str_gt = str_slus[b]

        hyp = tokenizer.decode(hyp_id, group_tokens=False, skip_special_tokens=True)
        target = tokenizer.decode(target_id, group_tokens=False, skip_special_tokens=True)

        if args.verbose:
            logger.info(f"SLU GT: {str_gt.split(' ')}")
            logger.info(f"SLU HYP: {hyp.split(' ')}")
            logger.info(f"SLU TARGET: {target.split(' ')}")
            logger.info(f"ASR HYP: {str_asr_hyp.split(' ')}")
            logger.info("-----")

        if hyp == target:
            correct_predictions += 1

        str_hyps.append(hyp)
        str_targets.append(target)

    # Compute metrics
    slu_wer = wer(str_targets, str_hyps)
    #asr_wer = wer(str_targets, str_asr_hyps)
    gt_wer = wer(str_targets, str_slus)
    accuracy = correct_predictions / valid_num_samples_to_use

    if epoch is None:
        epoch_info = ""
    else:
        epoch_info = f" at epoch {epoch}"
    logger.info(f"SLU WER{epoch_info}: {slu_wer * 100:.4f}%")
    logger.info(f"Ground Truth WER{epoch_info}: {gt_wer * 100:.4f}%")
    logger.info(f"Exact Matching{epoch_info}: {accuracy * 100:.4f}% ({correct_predictions}/{valid_num_samples_to_use})")

    intent_acc = None
    if getattr(args, 'use_self_prompt', False) and gt_intent_ids:
        valid_gt = sum(1 for g in gt_intent_ids if g >= 0)
        correct_intent = sum(1 for p, g in zip(pred_intent_ids, gt_intent_ids) if g >= 0 and p == g)
        intent_acc = correct_intent / max(1, valid_gt)
        logger.info(f"Intent Accuracy{epoch_info}: {intent_acc * 100:.4f}% ({correct_intent}/{valid_gt})")

    return {
        'slu_wer': slu_wer,
        #'asr_wer': asr_wer,
        'gt_wer': gt_wer,
        'accuracy': accuracy, # same meaning as exact matching, but added for clarity
        'em': accuracy,
        'total_samples': valid_num_samples_to_use,
        'correct_predictions': correct_predictions,
        'intent_acc': intent_acc,
    }


def compute_valid_loss(
    model,
    eval_dataset,
    tokenizer,
    args,
    sos_id,
    eos_id,
    device,
    epoch=None,
    criterion=nn.CrossEntropyLoss(reduction="none"),
):
    """
    Validation 수행 및 메트릭 계산
    매번 다른 샘플을 무작위로 선택하여 평가

    Args:
        model: 평가할 모델
        eval_dataset: 전체 evaluation dataset
        tokenizer: 토크나이저
        args: 학습 인자
        sos_id: 시작 토큰 ID
        eos_id: 종료 토큰 ID
        device: 학습 device
        epoch: 현재 epoch

    Returns:
        valid_loss 포함
    """
    device = (
        next(model.parameters()).device
        if not isinstance(model, torch.nn.DataParallel)
        else next(model.module.parameters()).device
    )
    device_str = str(device)
    use_cuda = "cuda" in device_str

    model.eval()

    # 전체 dataset에서 무작위로 num_samples만큼 선택
    total_samples = len(eval_dataset)
    val_dataset = eval_dataset
    logger.info(f"* Validation: Using all {total_samples:,} samples")

    # Validation용 DataLoader 생성
    val_sampler = BatchSampler(val_dataset,
                               batch_size=args.batch_size,
                               shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=hubert_and_deberta_dataset_collate_fn,
    )

    every_n = max(1, len(val_loader) // 4)

    tag_patterns = []
    if getattr(args, 'use_self_prompt', False):
        m = _get_base_model(model)
        prompt_table = getattr(m, 'prompt_table', [])
        tag_patterns = _build_tag_token_patterns(tokenizer, prompt_table)

    with torch.no_grad():
        step = 0
        count = 0
        loss_sum = 0.0
        ar_loss_sum = 0.0
        intent_loss_sum = 0.0
        intent_loss_count = 0
        for batch in val_loader:
            audio_feats = batch["feat"]
            audio_feat_mask = batch["feat_mask"]
            text_feats = batch["text_feat"]
            text_feat_mask = batch["text_mask"]
            slus = batch["slu"]
            slu_mask = batch["slu_mask"]

            audio_feats = audio_feats.to(device) # B, T, D
            audio_feat_mask = audio_feat_mask.to(device)
            text_feats = text_feats.to(device)
            text_feat_mask = text_feat_mask.to(device)

            audio_feats, audio_feat_mask, text_feats, text_feat_mask = apply_condition_type(
                args,
                audio_feats,
                audio_feat_mask,
                text_feats,
                text_feat_mask,
            )

            slus = slus.to(device)
            slu_mask = slu_mask.to(device)

            B = slus.size(0)
            lengths = slu_mask.sum(dim=1)  # B,
            T = lengths.max().item()  # max target length in the batch
            input_ids = torch.zeros((B, T+1), device=device, dtype=torch.long)
            input_ids[:, 1:] = slus
            input_ids[:, 0] = sos_id

            target_ids = torch.zeros((B, T+1), device=device, dtype=torch.long)
            target_ids[:, :-1] = slus
            target_ids[torch.arange(B), lengths] = eos_id

            input_mask = input_ids != 0  # B, T_o+1

            intent_gt = None
            prompt_prefix_lengths = None
            if getattr(args, 'use_self_prompt', False) and tag_patterns:
                intent_gt = _build_intent_gt_from_slu_ids(slus, slu_mask, tag_patterns)
                prompt_prefix_lengths = _build_prompt_prefix_lengths(intent_gt, prompt_table, tokenizer)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                # logits B, T, K
                out = model(input_ids=input_ids,
                            audio_feats=audio_feats,
                            audio_mask=audio_feat_mask,
                            text_feats=text_feats,
                            text_mask=text_feat_mask)
                if isinstance(out, tuple):
                    logits, prompt_logits = out
                else:
                    logits = out
                    prompt_logits = None

                logits_perm = logits.permute(0, -1, 1)
                ar_loss = criterion(logits_perm, target_ids)
                mask = input_mask.float()
                if getattr(args, 'use_self_prompt', False) and prompt_prefix_lengths is not None:
                    for b in range(B):
                        prefix_len = int(prompt_prefix_lengths[b].item())
                        if prefix_len > 0:
                            mask[b, :min(prefix_len, mask.size(1))] = 0.0
                denom = mask.sum().clamp_min(1.0)
                ar_loss = (ar_loss * mask).sum() / denom

                predict_intent_loss = None
                if prompt_logits is not None and intent_gt is not None:
                    intent_gt = intent_gt.to(prompt_logits.device, dtype=torch.long)
                    valid = intent_gt >= 0
                    if valid.any():
                        predict_intent_loss = F.cross_entropy(
                            prompt_logits[valid], intent_gt[valid]
                        )

                if predict_intent_loss is not None:
                    loss = ar_loss + args.intent_loss_weight * predict_intent_loss
                else:
                    loss = ar_loss

            loss_sum += loss.item()
            ar_loss_sum += ar_loss.item()
            if predict_intent_loss is not None:
                intent_loss_sum += predict_intent_loss.item()
                intent_loss_count += 1

            step += 1
            count += slus.size(0)
            """
            if step % every_n == 0:
                logger.info(f"Evaluation step {step:,}/{len(val_loader):,} completed.")
                logger.info(f"  Processed {count:,}/{total_samples:,} samples.")
            """
    valid_loss = loss_sum / max(1, step)
    valid_ar_loss = ar_loss_sum / max(1, step)
    valid_intent_loss = None
    if intent_loss_count > 0:
        valid_intent_loss = intent_loss_sum / intent_loss_count

    if epoch is None:
        epoch_info = ""
    else:
        epoch_info = f" at epoch {epoch}"
    log_msg = f"Valid loss{epoch_info}: total={valid_loss:.4f}, ar={valid_ar_loss:.4f}"
    if valid_intent_loss is not None:
        log_msg += f", predict_intent={valid_intent_loss:.4f}"
    logger.info(log_msg)

    return {
        'valid_loss': valid_loss,
        'valid_ar_loss': valid_ar_loss,
        'valid_intent_loss': valid_intent_loss,
    }


# -------------------------
# Train loop
# -------------------------
def train_model(
    args: ArgumentParser,
    model,
    train_loader,
    eval_dataset,
    optim,
    optim_scheduler,
    grad_clip: float = 1.0,
    sos_id: int = 1,
    eos_id: int = 2,
    tokenizer = None,
    augmentor = None,
    init_condition: Optional[Dict] = None,
):

    """
    if args.use_additional_loss_only:
        logger.warning("[WARNING] Using only additional loss for training.")
        logger.warning("n_step should be set to 2 for this setting.")
        args.n_step = 2
        logger.info(f"Setting args.n_step to {args.n_step}")
    """

    device = (
        next(model.parameters()).device
        if not isinstance(model, torch.nn.DataParallel)
        else next(model.module.parameters()).device
    )
    device_str = str(device)
    use_cuda = "cuda" in device_str
    scaler = GradScaler(enabled=use_cuda)

    criterion = nn.CrossEntropyLoss(reduction="none")

    loss_ema = None
    ema_beta = getattr(args, "loss_ema_beta", 0.98)

    step = init_condition.get("step", 1) if init_condition is not None else 1
    init_epoch = init_condition.get("epoch", 1) if init_condition is not None else 1

    best_valid_loss = float("inf")
    best_em = 0.0  # EM (exact matching) is higher is better

    # TAG token patterns: use_self_prompt 활성 시 predict_intent_loss 계산에 사용
    tag_patterns = []
    prompt_table = []
    if getattr(args, 'use_self_prompt', False):
        m = _get_base_model(model)
        prompt_table = getattr(m, 'prompt_table', [])
        tag_patterns = _build_tag_token_patterns(tokenizer, prompt_table)
        logger.info(f"* tag_patterns built: {len(tag_patterns)} labels")

    for epoch in range(init_epoch, args.final_epoch + 1):
        logger.info(f"===== Starting epoch {epoch} =====")

        # Training mode
        model.train()

        for batch in train_loader:
            audio_feats = batch["feat"]
            audio_feat_mask = batch["feat_mask"]
            text_feats = batch["text_feat"]
            text_feat_mask = batch["text_mask"]
            slus = batch["slu"]
            slu_mask = batch["slu_mask"]

            # x1: B, T_o
            # dtype/shape 정리
            audio_feats = audio_feats.to(device) # B, T, D
            audio_feat_mask = audio_feat_mask.to(device)
            text_feats = text_feats.to(device)
            text_feat_mask = text_feat_mask.to(device)
            slus = slus.to(device)
            slu_mask = slu_mask.to(device)

            if augmentor is not None:
                audio_feats, text_feats, audio_feat_mask, text_feat_mask = augmentor.apply(
                    audio_feats=audio_feats,
                    text_feats=text_feats,
                    audio_mask=audio_feat_mask,
                    text_mask=text_feat_mask,
                    step=step,
                )

            audio_feats, audio_feat_mask, text_feats, text_feat_mask = apply_condition_type(
                args,
                audio_feats,
                audio_feat_mask,
                text_feats,
                text_feat_mask,
            )

            B = slus.size(0)
            lengths = slu_mask.sum(dim=1)  # B,
            T = lengths.max().item()  # max target length in the batch
            input_ids = torch.zeros((B, T+1), device=device, dtype=torch.long)
            input_ids[:, 1:] = slus
            input_ids[:, 0] = sos_id

            target_ids = torch.zeros((B, T+1), device=device, dtype=torch.long)
            target_ids[:, :-1] = slus
            target_ids[torch.arange(B), lengths] = eos_id

            input_mask = input_ids != 0  # B, T_o+1

            # intent GT: SLU target token ids에서 TAG 패턴 매칭으로 추출
            intent_gt = None
            prompt_prefix_lengths = None
            if getattr(args, 'use_self_prompt', False) and tag_patterns:
                intent_gt = _build_intent_gt_from_slu_ids(slus, slu_mask, tag_patterns)
                prompt_prefix_lengths = _build_prompt_prefix_lengths(intent_gt, prompt_table, tokenizer)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                # logits B, T, K  (SelfPromptARModel은 (logits, prompt_logits) tuple 반환)
                out = model(input_ids=input_ids,
                            audio_feats=audio_feats,
                            audio_mask=audio_feat_mask,
                            text_feats=text_feats,
                            text_mask=text_feat_mask)
                
                if isinstance(out, tuple):                    
                    logits, prompt_logits = out
                else:
                    logits = out
                    prompt_logits = None

                logits_perm = logits.permute(0, -1, 1)
                ar_loss = criterion(logits_perm, target_ids)
                mask = input_mask.float()
                if getattr(args, 'use_self_prompt', False) and prompt_prefix_lengths is not None:
                    for b in range(B):
                        prefix_len = int(prompt_prefix_lengths[b].item())
                        if prefix_len > 0:
                            mask[b, :min(prefix_len, mask.size(1))] = 0.0
                denom = mask.sum().clamp_min(1.0)
                ar_loss = (ar_loss * mask).sum() / denom

                # predict_intent_loss: prompt_logits와 intent GT가 모두 있을 때만 CE loss
                predict_intent_loss = None
                if prompt_logits is not None and intent_gt is not None:
                    intent_gt = intent_gt.to(prompt_logits.device, dtype=torch.long)
                    valid = intent_gt >= 0
                    if valid.any():
                        predict_intent_loss = F.cross_entropy(
                            prompt_logits[valid], intent_gt[valid]
                        )
                    #print(f"{intent_gt[valid]=}")
                    #print(f"{prompt_logits[valid]=}")
                    #print(f"{intent_gt[valid].shape=}")
                    #print(f"{prompt_logits[valid].shape=}")
                    #print(f"{torch.argmax(prompt_logits[valid], dim=1)=}")
                    #import sys
                    #sys.exit(0)

            # Final loss combination
            if predict_intent_loss is not None:
                loss = ar_loss + args.intent_loss_weight * predict_intent_loss
            else:
                loss = ar_loss

            optim.zero_grad(set_to_none=True)
            prev_scale = scaler.get_scale()
            scaler.scale(loss).backward()

            grad_norm = None
            if grad_clip is not None:
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip
                )
            scaler.step(optim)
            scaler.update()

            if scaler.get_scale() >= prev_scale:
                optim_scheduler.step()

            loss_val = float(loss.detach().cpu())
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * loss_val

            if step % args.log_step == 0:
                grad_norm_val = float(grad_norm) if grad_norm is not None else 0.0
                intent_loss_val = predict_intent_loss.item() if predict_intent_loss is not None else 0.0
                log_msg = (
                    f"[Epoch {epoch}] "
                    f"[step {step:,}] "
                    f"lr={optim_scheduler.get_last_lr()[0]:.6f}, "
                    f"loss={loss_val:.6f}, "
                    f"loss_ema={loss_ema:.6f}, "
                    f"ar_loss={ar_loss.item():.6f}, "
                )
                if getattr(args, 'use_self_prompt', False):
                    log_msg += f"predict_intent_loss={intent_loss_val:.6f}, "
                log_msg += (
                    f"grad_norm={grad_norm_val:.4f}, "
                    f"scale={scaler.get_scale():.1f}"
                )
                logger.info(log_msg)
            # Increment step counter
            step += 1

        # End of epoch
        if args.make_model_dir:
            ckpt_path = os.path.join(save_dir, "model", f"model_epoch{epoch}.pt")
        else:
            ckpt_path = os.path.join(save_dir, f"model_epoch{epoch}.pt")
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
            },
            ckpt_path,
        )
        logger.info(f"Saved: {ckpt_path}")

        # check validation (validation loss only)
        if epoch % args.eval_epoch == 0:
            logger.info(f"===== Valid Loss Calculation at epoch {epoch} =====")
            valid_loss = compute_valid_loss(
                model=model,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                args=args,
                sos_id=sos_id,
                eos_id=eos_id,
                device=device,
                epoch=epoch,
                criterion=criterion,
            )

            # Update best model based on valid loss (lower is better)
            if update_best_model_link(
                metric_value=valid_loss['valid_loss'],
                best_metric_value=best_valid_loss,
                metric_name="valid_loss",
                is_better_fn=lambda x, y: x < y,  # lower loss is better
                save_dir=save_dir,
                ckpt_path=ckpt_path,
                args=args,
            ):
                best_valid_loss = valid_loss['valid_loss']

        # Run validation (running inference) with random sampling
        if epoch % args.eval_epoch == 0:
            logger.info(f"===== Sampled validation at epoch {epoch} =====")
            validate_result = validate_model(
                model=model,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                args=args,
                sos_id=sos_id,
                eos_id=eos_id,
                device=device,
                valid_num_samples=args.valid_num_samples,
                epoch=epoch,
            )

            # Update best model based on EM (higher is better)
            if update_best_model_link(
                metric_value=validate_result['em'],
                best_metric_value=best_em,
                metric_name="em",
                is_better_fn=lambda x, y: x > y,  # higher EM is better
                save_dir=save_dir,
                ckpt_path=ckpt_path,
                args=args,
            ):
                best_em = validate_result['em']

    return


if __name__ == "__main__":

    args = build_parser().parse_args()

    # remove save_dir if reset_save_dir is True
    if args.reset_save_dir and os.path.exists(args.save_dir):
        import shutil
        existing_ckpts = (
            glob.glob(os.path.join(args.save_dir, "model_*.pt")) +
            glob.glob(os.path.join(args.save_dir, "model", "model_*.pt"))
        )
        if existing_ckpts:
            print(f"[WARNING] Found {len(existing_ckpts)} checkpoint(s) in '{args.save_dir}':")
            for p in sorted(existing_ckpts):
                print(f"  {p}")
            answer = input("Delete save_dir and all checkpoints? [y/N] ").strip().lower()
            if answer == "y":
                shutil.rmtree(args.save_dir)
                print(f"Deleted: {args.save_dir}")
            else:
                print("Aborted. Keeping existing save_dir.")
                args.reset_save_dir = False
        else:
            shutil.rmtree(args.save_dir)

    if args.save_dir is None:
        args.save_dir = "garbage"
    else:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

    if args.make_model_dir:
        os.makedirs(os.path.join(args.save_dir, "model"), exist_ok=True)

    setup_logger(args.save_dir, log_name="train")
    logger = logging.getLogger()

    # Save command line
    logger.info(f"* Command Line: {' '.join(sys.argv)}")
    logger.info(f"* Configuration")
    logger.info(json.dumps(vars(args), indent=2))
    """
    # seed (필요하면)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # tokenizer
    processor = AutoProcessor.from_pretrained(args.tokenizer_model_name)
    vocab_size_before = len(processor.tokenizer)
    # adding numbers from 0 to 9 + "[MASK]" if not already present
    additional_tokens = ["[", "]", ":", "_"]
    new_tokens = [str(i) for i in range(10)] + [args.mask_token] + additional_tokens
    num_base_added = processor.tokenizer.add_tokens(new_tokens)
    num_intent_added = 0
    intent_tokens = []

    # Optional: add intent-related tokens from INTENT file.
    if args.use_intent_token:
        intent_labels = _load_intent_labels(args.prompt_tag_path)        
        intent_tokens = _build_intent_token_candidates(intent_labels)        
        num_intent_added = processor.tokenizer.add_tokens(intent_tokens) if intent_tokens else 0

    logger.info(f"{num_base_added} base tokens added to the tokenizer.")
    if args.use_intent_token:
        logger.info(
            f"{num_intent_added} intent-related tokens added to the tokenizer "
            f"(candidates={len(intent_tokens)})."
        )
    else:
        logger.info("Intent-token injection disabled (use_intent_token=False).")
    tokenizer = processor.tokenizer

    # Always sync model vocab size with tokenizer after token injection.
    actual_vocab_size = len(tokenizer)
    expected_vocab_size = vocab_size_before + num_base_added + num_intent_added
    logger.info(
        f"Tokenizer vocab size: {vocab_size_before} + {num_base_added} (base) + "
        f"{num_intent_added} (intent) = {actual_vocab_size}"
    )
    if actual_vocab_size != expected_vocab_size:
        logger.warning(
            f"Tokenizer size check mismatch: expected {expected_vocab_size}, got {actual_vocab_size}."
        )
    if actual_vocab_size != args.vocab_size:
        logger.warning(
            f"vocab_size argument ({args.vocab_size}) does not match tokenizer vocab size "
            f"({actual_vocab_size}). Overriding args.vocab_size."
        )
    args.vocab_size = actual_vocab_size

    if args.use_self_prompt:
        cfg = SelfPromptARModelConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            audio_dim=args.audio_dim,
            text_dim=args.text_dim,
            max_output_length=args.max_output_length,
            model_type=args.model_type,
            norm_first=args.norm_first,
            num_intent=args.num_intent,
            length_hidden_dim=args.length_hidden_dim,
            length_dropout=args.length_dropout,
            prompt_tag_path=args.prompt_tag_path,
        )
        logger.info(f"* SelfPromptARModelConfig: ")
        logger.info(json.dumps(asdict(cfg), indent=2))
        model = SelfPromptARModel(cfg)
        loaded_from_file = getattr(model, "prompt_table_loaded_from_file", False)
        loaded_path = getattr(model, "prompt_table_path", args.prompt_tag_path)
        if loaded_from_file:
            logger.info(f"* Loaded intent prompt table from: {loaded_path}")
        else:
            logger.warning(
                f"* Intent prompt file not found at resolved path: {loaded_path}. "
                "Using synthetic fallback labels (INTENT_0..)."
            )
    else:
        cfg = ARModelConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            audio_dim=args.audio_dim,
            text_dim=args.text_dim,
            max_output_length=args.max_output_length,
            model_type=args.model_type,
            norm_first=args.norm_first,
        )
        logger.info(f"* ARModelConfig: ")
        logger.info(json.dumps(asdict(cfg), indent=2))
        model = ARModel(cfg)

    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        state_dict = remove_module_prefix(checkpoint["model"])
        model.load_state_dict(state_dict)
        logger.info(f"* Loaded checkpoint from {args.ckpt_path}")

    #logger.info(f"{dfm_model.device=}")
    trainable_params = 0
    model_parts = [model.slu_model]
    if args.use_self_prompt:
        model_parts.append(model.prompt_predictor)
    for model_part in model_parts:
        tmp_trainable_params = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        trainable_params += tmp_trainable_params
        logger.info(f"{class_name(model_part)} Trainable Parameters: {tmp_trainable_params:,}")
    logger.info(f"* Total Trainable Parameters: {trainable_params:,}")

    # 멀티 GPU 지원
    primary_gpu_id = 0
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        gpu_ids = [int(x) for x in args.gpu.split(",")]
        num_gpus = len(gpu_ids)
        logger.info(f"{torch.cuda.device_count()} GPUs are available")
        logger.info(f"GPU option: '{args.gpu}'")
        logger.info(f"Using {num_gpus} GPUs (IDs: {gpu_ids})")
        if num_gpus > 1:
            primary_gpu_id = gpu_ids[0]
            device = torch.device(f"cuda:{primary_gpu_id}")
            model = model.to(device)
            model = torch.nn.DataParallel(model,
                                          device_ids=gpu_ids,
                                          output_device=primary_gpu_id)
        else:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            model = model.to(device)
            #logger.info(f"{model.device=}")
            logger.info(f"Using single GPU: cuda:{gpu_ids[0]}")
    else:
        if device.type == "cuda":
            gpu_ids = [int(x) for x in args.gpu.split(",")]
            device = torch.device(f"cuda:{gpu_ids[0]}")
            model = model.to(device)
            logger.info(f"Using single GPU: {device}")
        else:
            model = model.to(device)
            logger.info("Using CPU device")
    """
    # Device 정보 확인
    logger.info(f"Device type: {device.type}")
    logger.info(f"Device index: {device.index}")
    logger.info(f"Device string: {str(device)}")
    logger.info(f"Model device: {dfm_model.device}")
    """
    #import sys; sys.exit(0)

    optim = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    def lr_lambda(current_step):
        if current_step < args.warmup_step:
            return float(current_step) / float(max(1, args.warmup_step))

        progress = (
            float(current_step - args.warmup_step) /
            float(max(1, args.total_step - args.warmup_step))
        )

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    optim_scheduler = LambdaLR(optim, lr_lambda)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    init_condition = {}
    if args.ckpt_path is not None:
        optim.load_state_dict(checkpoint["optim"])
        scaler.load_state_dict(checkpoint["scaler"])
        init_condition["step"] = checkpoint.get("step", 0) + 1
        init_condition["epoch"] = checkpoint.get("epoch", 0) + 1
        logger.info(f"* Loaded optimizer and scaler states from {args.ckpt_path}, "
                    f"resuming from step {init_condition['step']}, "
                    f"epoch {init_condition['epoch']}. ")

    train_dataset = HuBERTandDeBERTaDataset(
        task=args.train_task,
        tokenizer=tokenizer,
        feat_dir=args.dataset_path,
        debugging=args.dataset_debugging,
        debugging_num=args.dataset_debugging_num,
        use_tar=args.use_tar,
    )

    train_sampler = BatchSampler(train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    train_dl = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=hubert_and_deberta_dataset_collate_fn,
    )

    total_train_data_count = len(train_dl.dataset)
    logger.info(f"* number of total train data: {total_train_data_count:,}")

    # Evaluation dataset (전체 로드, validation에서 매번 무작위 샘플링)
    eval_dataset = HuBERTandDeBERTaDataset(
        task=args.eval_task,
        tokenizer=tokenizer,
        feat_dir=args.dataset_path,
        debugging=args.dataset_debugging,
        debugging_num=args.dataset_debugging_num,
        use_tar=args.use_tar,
    )

    total_eval_samples = len(eval_dataset)
    logger.info(f"* number of total eval data: {total_eval_samples:,}")
    logger.info(f"* Sampling {min(args.valid_num_samples, total_eval_samples):,} samples randomly for each validation.")

    sos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    mask_id = tokenizer.convert_tokens_to_ids(args.mask_token)
    logger.info(f"* ID of {args.mask_token}: {mask_id}")

    augmentor = build_augmentor(args)
    logger.info(f"* Augmentor: {class_name(augmentor)}")

    train_model(
        args=args,
        model=model,
        train_loader=train_dl,
        eval_dataset=eval_dataset,
        optim=optim,
        optim_scheduler=optim_scheduler,
        sos_id=sos_id,
        eos_id=eos_id,
        tokenizer=tokenizer,
        augmentor=augmentor,
        init_condition=init_condition,
    )