import os
import sys
import argparse
from argparse import ArgumentParser
from typing import Optional, Dict
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

# For DFML
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver

from model import DFMModel, DFMModelConfig, DFMModelWrapper
from hubert_deberta_dataset import HuBERTandDeBERTaDataset
from hubert_deberta_dataset import hubert_and_deberta_dataset_collate_fn
from hubert_deberta_dataset import BatchSampler
from sampling import sampling_batch, sampling_debugging
from custom_path import UniformDiscreteProbPath

from jiwer import wer
from utils import str2bool, remove_module_prefix
from utils import setup_logger, class_name


def build_parser():
    p = argparse.ArgumentParser(description="Train DFM (based on DiT) with HuBERT + DeBERTa features")

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
    p.add_argument("--uniform", type=str2bool, default=False)
    p.add_argument("--loss_type", type=str,
                   default="ce", choices=["ce", "gkl"], help="ce or gkl(generalized KL)")    
    #p.add_argument("--use_additional_loss_only", action="store_true", default=False,
    #               help="For comparison, use only additional loss term")
    p.add_argument("--seed", type=int, default=42)    
    p.add_argument("--ckpt_path", type=str, default=None, help="Path to load checkpoint")
    p.add_argument("--gpu", type=str, default="0", help="GPU ids separated by comma, e.g., '0,1,2'")
    p.add_argument("--use_tar", type=str2bool, default=True, help="Whether to use tarred dataset")
    # additional loss
    p.add_argument("--alpha", type=float, default=0.1, help="Weight for additional loss term")    
    p.add_argument("--use_additional_loss", type=str2bool, default=False,
                   help="Whether to use additional loss term")

    # ---- model dims / arch ----
    ## for DFM model
    p.add_argument("--vocab_size", type=int, default=43)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--audio_dim", type=int, default=1024)
    p.add_argument("--text_dim", type=int, default=1024)
    p.add_argument("--max_output_length", type=int, default=256, help="Maximum output length during inference")
    p.add_argument("--noise_ratio", type=float, default=0.5, help="Noise ratio for UniformDiscreteProbPath")
    p.add_argument("--n_step", type=int, default=5, help="Number of sampling steps during inference")    
    p.add_argument("--model_type", type=str, choices=["dit", "transformer"], default="dit")
    
    ## for length predictor
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--length_hidden_dim", type=int, default=512)
    #p.add_argument("--max_target_positions", type=int, default=256)
    p.add_argument("--length_dropout", type=float, default=0.1)
    p.add_argument("--length_condition", type=str, choices=["audio", "text", "both"], default="text")
    p.add_argument("--length_margin", type=float, default=0.1)
    p.add_argument("--length_loss_weight", type=float, default=1.0)
    

    # ---- data / tokenization ----, 
    p.add_argument("--dataset_path", type=str, default="./hubert_deberta_cache_retrial")
    p.add_argument("--train_task", type=str, default="train")
    p.add_argument("--eval_task", type=str, default="eval")
    p.add_argument("--test_task", type=str, default="test")
    p.add_argument("--tokenizer_model_name", type=str, default="facebook/hubert-large-ls960-ft")
    p.add_argument("--mask_token", type=str, default="[MASK]")
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

    return p


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
    mask_id,
    device,
    valid_num_samples=2048,
    debugging=False,
):
    """
    Validation 수행 및 메트릭 계산
    매번 다른 샘플을 무작위로 선택하여 평가
    
    Args:
        dfm_model: 평가할 모델
        eval_dataset: 전체 evaluation dataset
        tokenizer: 토크나이저
        args: 학습 인자
        mask_id: 마스크 토큰 ID
        device: 학습 device
        valid_num_samples: validation에 사용할 샘플 개수 (기본값: 2048)
        debugging: 디버깅 모드
    
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
    wrapper = DFMModelWrapper(model)
    # a convex path path
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = UniformDiscreteProbPath(scheduler=scheduler,
                                   vocab_size=args.vocab_size,
                                   noise_ratio=args.noise_ratio)

    solver = MixtureDiscreteEulerSolver(
        model=wrapper,
        path=path,
        vocabulary_size=args.vocab_size,
    )
    eps = 1e-4
    step_size = 0.01
    n_step = args.n_step
    time_grid = torch.linspace(0.0, 1.0 - eps, n_step, device=device)

    hyp_ids = []
    target_ids = []
    str_asr_hyps = []
    str_sc_targets = []
    step = 0
    count = 0
    for batch in val_loader:
        # batch
        (
            audio_feats, audio_feat_mask,
            text_feats, text_feat_mask,
            gts, hyps,
            gt_mask, hyp_mask,
            str_gts, str_hyps,
        ) = batch

        audio_feats = audio_feats.to(device) # B, T, D
        audio_feat_mask = audio_feat_mask.to(device)
        text_feats = text_feats.to(device)
        text_feat_mask = text_feat_mask.to(device)

        with torch.no_grad():
            predicted_lengths = model.predict_lengths(
                text_feats, text_feat_mask.bool()
            )
        original_hyp_lengths = hyp_mask.sum(dim=1).to(device)  # B,

        B = audio_feats.size(0)
        T = max(predicted_lengths.max().item(), original_hyp_lengths.max().item())        

        x_0 = torch.zeros((B, T), device=device, dtype=torch.long)        
        pos_idx = torch.arange(T, device=device).unsqueeze(0)     # 1, T
        hyp_mask = pos_idx < original_hyp_lengths.unsqueeze(1)    # B, T
        hyps_padded = torch.full((B, T), mask_id, device=device, dtype=torch.long)
        hyps_padded[:, :hyps.size(1)] = hyps
        x_0[hyp_mask] = hyps_padded[hyp_mask]

        with torch.no_grad():
            x_1_hat = solver.sample(
                x_init=x_0,
                step_size=step_size,
                time_grid=time_grid,
                return_intermediates=True,            
                audio_feats=audio_feats,
                audio_mask=audio_feat_mask,
                text_feats=text_feats,
                text_mask=text_feat_mask,
            )

        x_1_hat[:, ~hyp_mask] = 0 #blank token for padding positionsq
        hyp_ids.extend(x_1_hat[-1].cpu().tolist())
        target_ids.extend(gts.tolist())
        str_asr_hyps.extend(str_hyps)
        str_sc_targets.extend(str_gts)

        step += 1
        count += B
        
        if debugging:
            logger.info(f"{gts=}")
            logger.info(f"{x_1_hat[-1]=}")
            logger.info(f"{gts.shape=}")
            logger.info(f"{x_1_hat[-1].shape=}")

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
        str_gt = str_sc_targets[b]
        
        hyp = tokenizer.decode(hyp_id, group_tokens=False, skip_special_tokens=True)
        target = tokenizer.decode(target_id, group_tokens=False, skip_special_tokens=True)        
        
        if args.verbose:
            logger.info(f"GT: {str_gt.split(' ')}")
            logger.info(f"TARGET: {target.split(' ')}")
            logger.info(f"ASR HYP: {str_asr_hyp.split(' ')}")
            logger.info(f"DFM HYP: {hyp.split(' ')}")
            logger.info("-----")
        
        if hyp == target:
            correct_predictions += 1
        
        str_hyps.append(hyp)
        str_targets.append(target)
    
    # Compute metrics
    dfm_wer = wer(str_targets, str_hyps)
    asr_wer = wer(str_targets, str_asr_hyps)
    gt_wer = wer(str_targets, str_sc_targets)
    accuracy = correct_predictions / valid_num_samples_to_use
    
    logger.info(f"DFM WER: {dfm_wer * 100:.4f}%")
    logger.info(f"ASR WER: {asr_wer * 100:.4f}%")
    logger.info(f"Ground Truth WER: {gt_wer * 100:.4f}%")
    logger.info(f"Exact Matching: {accuracy * 100:.4f}% ({correct_predictions}/{valid_num_samples_to_use})")
    
    return {
        'dfm_wer': dfm_wer,
        'asr_wer': asr_wer,
        'gt_wer': gt_wer,
        'accuracy': accuracy,
        'total_samples': valid_num_samples_to_use,
        'correct_predictions': correct_predictions,
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
    mask_id = 1,
    tokenizer = None,
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
    
    scheduler = PolynomialConvexScheduler(n=2.0)    
    path = UniformDiscreteProbPath(scheduler=scheduler,
                                   vocab_size=args.vocab_size,
                                   noise_ratio=args.noise_ratio)

    length_criterion = nn.CrossEntropyLoss(reduction="mean")
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.loss_type == "gkl":
        criterion = MixturePathGeneralizedKL(path)
    else:
        raise ValueError(
            f"Unknown loss_type: {args.loss_type}. "
            "Supported loss types are: ['ce', 'gkl']"
        )
    
    loss_ema = None
    ema_beta = getattr(args, "loss_ema_beta", 0.98)
    
    step = init_condition.get("step", 1) if init_condition is not None else 1
    init_epoch = init_condition.get("epoch", 1) if init_condition is not None else 1

    for epoch in range(init_epoch, args.final_epoch + 1):        
        logger.info(f"===== Starting epoch {epoch} =====")
        
        # Training mode
        model.train()
        
        for batch in train_loader:
            # batch
            (
                audio_feats, audio_feat_mask,
                text_feats, text_feat_mask,
                gts, hyps,
                gt_mask, hyp_mask,
                str_gts, str_hyps,
            ) = batch

            # x1: B, T_o
            # dtype/shape 정리
            audio_feats = audio_feats.to(device) # B, T, D
            audio_feat_mask = audio_feat_mask.to(device)
            text_feats = text_feats.to(device)
            text_feat_mask = text_feat_mask.to(device)
            #x1 = gts.to(device)
            #x1_mask = gt_mask.to(device)
                        
            optim.zero_grad(set_to_none=True)
                        
            # length prediction
            original_lengths = gt_mask.sum(dim=1).to(device)  # B,
            original_hyp_lengths = hyp_mask.sum(dim=1).to(device)  # B,
            
            # predicted lengths = original length * ( 1 + margin )
            lengths = original_lengths * (1.0 + args.length_margin) 
            lengths = lengths.float().ceil().long()
            lengths = torch.clamp(lengths, min=1, max=args.max_output_length)
            K = args.vocab_size
            B = gts.size(0)            
            T = int(max(lengths.max().item(), original_hyp_lengths.max().item()))
            
            # Vectorized approach: much faster than for loops
            x0 = torch.zeros((B, T), device=device, dtype=torch.long)
            x1 = torch.zeros((B, T), dtype=torch.long, device=device)
            
            # Create position indices for vectorized indexing
            batch_idx = torch.arange(B, device=device).unsqueeze(1)  # B, 1
            pos_idx = torch.arange(T, device=device).unsqueeze(0)     # 1, T
            
            # For x0: copy from hyps where position < original_hyp_lengths
            hyp_mask = pos_idx < original_hyp_lengths.unsqueeze(1)    # B, T
            hyps_padded = torch.full((B, T), mask_id, device=device, dtype=torch.long)
            hyps_padded[:, :hyps.size(1)] = hyps
            x0[hyp_mask] = hyps_padded[hyp_mask]
            
            # For x1: copy from gts where position < original_lengths
            gt_mask = pos_idx < original_lengths.unsqueeze(1)         # B, T
            gts_padded = torch.zeros((B, T), device=device, dtype=torch.long)
            gts_padded[:, :gts.size(1)] = gts
            x1[gt_mask] = gts_padded[gt_mask]

            # sample time t ~ Uniform(eps,1)
            t = torch.rand(B, device=device).clamp(1e-4, 1.0 - 1e-4)

            sample = path.sample(t=t, x_0=x0, x_1=x1)
            
            with torch.amp.autocast('cuda', enabled=False):
                # logits B, T, K
                logits = model(x_t=sample.x_t,
                               t=sample.t,
                               audio_feats=audio_feats,
                               audio_mask=audio_feat_mask,
                               text_feats=text_feats,
                               text_mask=text_feat_mask)

                if args.loss_type == "ce":
                    corrupt_mask = (x1 != sample.x_t)  # B, T_out
                    logits_perm = logits.permute(0, -1, 1)
                    dfm_loss = criterion(logits_perm, x1)
                    mask = corrupt_mask.float()
                    denom = mask.sum().clamp_min(1.0)
                    dfm_loss = (dfm_loss * mask).sum() / denom
                elif args.loss_type == "gkl":
                    dfm_loss = criterion(logits=logits,
                                        x_1=sample.x_1,
                                        x_t=sample.x_t,
                                        t=sample.t)

                # Additional loss (skip if not enabled)
                additional_loss = torch.tensor(0.0, device=device)
                if args.use_additional_loss:
                    logits_perm = logits.permute(0, -1, 1)  # B, K, T
                    additional_loss = criterion(logits_perm, x1).mean()                    

                # for length loss
                if args.length_condition == "text":
                    length_logits = model.predict_length_logits(
                        text_feats, text_feat_mask.bool()
                    )
                elif args.length_condition == "audio":
                    length_logits = model.predict_length_logits(
                        audio_feats, audio_feat_mask.bool()
                    )
                length_loss = length_criterion(length_logits, original_lengths)

            # Final loss combination
            loss = ( dfm_loss + args.alpha * additional_loss 
                    + args.length_loss_weight * length_loss
            )

            prev_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
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
                logger.info(f"[Epoch {epoch}] "
                        f"[step {step:,}] "
                        f"lr={optim_scheduler.get_last_lr()[0]:.6f}, "
                        f"loss={loss_val:.4f}, "
                        f"loss_ema={loss_ema:.4f}, "
                        f"dfm_loss={float(dfm_loss.detach().cpu()):.4f}, "
                        f"length_loss={float(length_loss.detach().cpu()):.4f}, "
                        f"length_weight={args.length_loss_weight}, "
                        f"additional_loss={float(additional_loss.detach().cpu()):.4f}, "                        
                        f"alpha={args.alpha}, "
                ) 

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
                "K": K,
            },
            ckpt_path,
        )
        logger.info(f"Saved: {ckpt_path}")
        
        # Run validation with random sampling
        if epoch % args.eval_epoch == 0:
            logger.info(f"===== Validation at epoch {epoch} =====")
            validate_model(
                model=model,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                args=args,
                mask_id=mask_id,
                device=device,
                valid_num_samples=args.valid_num_samples,
                debugging=args.debugging,
            )        

    return


if __name__ == "__main__":

    args = build_parser().parse_args()

    # remove save_dir if reset_save_dir is True
    if args.reset_save_dir and os.path.exists(args.save_dir):        
        import shutil
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
    # adding numbers from 0 to 9 + "[MASK]" if not already present
    new_tokens = [str(i) for i in range(10)] + [args.mask_token]
    num_added = processor.tokenizer.add_tokens(new_tokens)
    logger.info(f"{num_added} tokens added to the tokenizer.")
    tokenizer = processor.tokenizer

    # num tokens 확인
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != args.vocab_size:
        logger.warning(f"vocab_size argument ({args.vocab_size}) "
              f"does not match actual tokenizer vocab size ({actual_vocab_size}). "
              f"Using actual vocab size.")
        args.vocab_size = actual_vocab_size

    cfg = DFMModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        audio_dim=args.audio_dim,
        text_dim=args.text_dim, 
        max_output_length=args.max_output_length,
        n_step=args.n_step,           
        model_type=args.model_type,
        
        # for length predictor
        embed_dim=args.embed_dim,
        length_hidden_dim=args.length_hidden_dim,
        max_target_positions=args.max_output_length,    
        length_dropout=args.length_dropout,
        length_condition=args.length_condition,
        length_margin=args.length_margin,
    )

    logger.info(f"* DFMConfig: ")
    logger.info(json.dumps(asdict(cfg), indent=2))
    
    #dfm_model = DFMModel(cfg, device=device)
    model = DFMModel(cfg)
    
    if args.ckpt_path is not None:        
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        state_dict = remove_module_prefix(checkpoint["model"])
        model.load_state_dict(state_dict)        
        logger.info(f"* Loaded checkpoint from {args.ckpt_path}")

    #logger.info(f"{dfm_model.device=}")
    trainable_params = 0
    for model_part in [model.dfm_model, model.length_predictor]:
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

    mask_id = tokenizer.convert_tokens_to_ids(args.mask_token)
    logger.info(f"* ID of {args.mask_token}: {mask_id}")

    train_model(
        args=args,
        model=model,
        train_loader=train_dl,
        eval_dataset=eval_dataset,
        optim=optim,
        optim_scheduler=optim_scheduler,
        mask_id=mask_id,
        tokenizer=tokenizer,        
        init_condition=init_condition,        
    )