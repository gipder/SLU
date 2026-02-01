import os
import sys
import argparse
from argparse import ArgumentParser
from typing import Optional
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

from jiwer import wer
from utils import str2bool, remove_module_prefix
from utils import setup_logger




def build_parser():
    p = argparse.ArgumentParser(description="Train DFM (based on U-Net) with HuBERT + DeBERTa features")

    # ---- training ----
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--test_batch_size", type=int, default=512)
    p.add_argument("--total_step", type=int, default=800000)
    p.add_argument("--log_step", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate by warmup step")
    p.add_argument("--warmup_step", type=int, default=2500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default=None, help="If None, auto-generated from lr.")
    p.add_argument("--save_step", type=int, default=50000)
    p.add_argument("--uniform", type=str2bool, default=False)
    p.add_argument("--loss_type", type=str,
                   default="ce", choices=["ce", "gkl"], help="ce or gkl(generalized KL)")
    p.add_argument("--use_additional_loss", type=str2bool, default=False,
                   help="Whether to use additional loss term")
    p.add_argument("--use_additional_loss_only", action="store_true", default=False,
                   help="For comparison, use only additional loss term")
    p.add_argument("--seed", type=int, default=42)    
    p.add_argument("--ckpt_path", type=str, default=None, help="Path to load checkpoint")
    p.add_argument("--gpu", type=str, default="0", help="GPU ids separated by comma, e.g., '0,1,2'")
    p.add_argument("--use_tar", type=str2bool, default=True, help="Whether to use tarred dataset")

    # ---- model dims / arch ----
    ## for DiT model
    p.add_argument("--vocab_size", type=int, default=43)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--audio_dim", type=int, default=1024)
    p.add_argument("--text_dim", type=int, default=1024)
    p.add_argument("--max_output_length", type=int, default=256, help="Maximum output length during inference")
    p.add_argument("--n_step", type=int, default=4, help="Number of sampling steps during inference")
    # additional loss
    p.add_argument("--alpha", type=float, default=0.1, help="Weight for additional loss term")    
    ## for length predictor
    #p.add_argument("--embed_dim", type=int, default=1024)
    #p.add_argument("--length_hidden_dim", type=int, default=512)
    #p.add_argument("--max_target_positions", type=int, default=256)
    #p.add_argument("--length_dropout", type=float, default=0.1)
    #p.add_argument("--length_condition", type=str, choices=["audio", "text"], default="text")

    # ---- data / tokenization ----, 
    p.add_argument("--dataset_path", type=str, default="./hubert_deberta_cache_retrial")
    p.add_argument("--train_task", type=str, default="train")
    p.add_argument("--eval_task", type=str, default="eval")
    p.add_argument("--test_task", type=str, default="test")
    p.add_argument("--tokenizer_model_name", type=str, default="facebook/hubert-large-ls960-ft")
    p.add_argument("--mask_token", type=str, default="[MASK]")
    #p.add_argument("--shuffle_train", type=bool, default=True)    

    # ---- debugging ----
    # debugging for dataset
    p.add_argument("--dataset_debugging", action="store_true", help="Enable debugging mode for dataset")    
    p.add_argument("--dataset_debugging_num", type=int, default=128, help="How many samples are used in debugging dataset")
    # debugging for sampling and others...
    p.add_argument("--debugging", action="store_true", help="Enable debugging mode in train_dfm()")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging during sampling")

    # ---- device ----
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    return p


# -------------------------
# Train loop
# -------------------------
def train_dfm(
    args: ArgumentParser,
    dfm_model,    
    train_loader,
    eval_loader,
    optim,
    optim_scheduler,    
    grad_clip: float = 1.0,
    use_amp: bool = True,
    mask_id = 1,
    init_step: int = 0,
    debugging: bool = False,    
):
    if args.save_dir is None:
        logger.info(f"Save dir: {args.save_dir}")
        logger.info(f"No specified save_dir")
        logger.info(f"save_dir will be 'garbage'")
        save_dir = "garbage"
    else:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

    if args.use_additional_loss_only:
        logger.warning("[WARNING] Using only additional loss for training.")
        logger.warning("n_step should be set to 2 for this setting.")
        args.n_step = 2
        logger.info(f"Setting args.n_step to {args.n_step}")

    device = (
        dfm_model.device 
        if not isinstance(dfm_model, torch.nn.DataParallel) 
        else dfm_model.module.device
    )
    device_str = str(device)
    use_cuda = "cuda" in device_str
    scaler = GradScaler(enabled=use_cuda)
    #sp = train_loader.dataset.sp
    # a convex path path
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

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
    
    train_iter = iter(train_loader)    
    for step in range(init_step, args.total_step):
        # train mode
        dfm_model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
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
        x1 = gts.to(device)
        x1_mask = gt_mask.to(device)
        
        K = args.vocab_size
        B = x1.size(0)
        T = args.max_output_length
        
        optim.zero_grad(set_to_none=True)
        
        if not args.use_additional_loss_only:
            # sample time t ~ Uniform(eps,1)
            t = torch.rand(B, device=device).clamp(1e-4, 1.0 - 1e-4)
        else:
            t = torch.zeros(B, device=device)

        if args.uniform:
            # x0 ~ Uniform(0, K-1) , (B, T_out, K)
            x0 = torch.randint(0, K, (B, T), device=device)
        else:
            # x0 ~ Mask(0, K-1)
            x0 = torch.full_like(x1, mask_id, device=device)

        sample = path.sample(t=t, x_0=x0, x_1=x1)
        
        with torch.amp.autocast('cuda', enabled=False):
            # logits B, T, K
            logits = dfm_model(x_t=sample.x_t,
                               t=sample.t,
                               audio_feats=audio_feats,
                               audio_mask=audio_feat_mask,
                               text_feats=text_feats,
                               text_mask=text_feat_mask)
            
            # DFM loss (skip if using additional_loss_only)
            if not args.use_additional_loss_only:
                if args.loss_type == "ce":
                    corrupt_mask = (x1 != sample.x_t)  # B, T_out
                    logits_perm = logits.permute(0, -1, 1)
                    dit_loss = criterion(logits_perm, x1)
                    mask = corrupt_mask.float()
                    denom = mask.sum().clamp_min(1.0)
                    dit_loss = (dit_loss * mask).sum() / denom
                elif args.loss_type == "gkl":
                    dit_loss = criterion(logits=logits,
                                         x_1=sample.x_1,
                                         x_t=sample.x_t,
                                         t=sample.t)
            else:
                dit_loss = torch.tensor(0.0, device=device)

            # Additional loss (skip if not enabled)
            additional_loss = torch.tensor(0.0, device=device)
            if args.use_additional_loss or args.use_additional_loss_only:
                logits_perm = logits.permute(0, -1, 1)  # B, K, T
                additional_loss = F.cross_entropy(
                    logits_perm, x1, reduction="mean"
                )

        # Final loss combination
        loss = dit_loss + args.alpha * additional_loss

        prev_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                dfm_model.parameters(),
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
            logger.info(f"[step {step:,}] "
                    f"loss={loss_val:.4f}, "
                    f"loss_ema={loss_ema:.4f}, "
                    f"dit_loss={float(dit_loss.detach().cpu()):.4f}, "
                    f"additional_loss={float(additional_loss.detach().cpu()):.4f}, "
                    f"lr={optim_scheduler.get_last_lr()[0]:.6f}, "
                    f"alpha={args.alpha} "
            )
            #break # for debugging
        
        if step % args.save_step == 0:
            # save each epoch
            ckpt_path = os.path.join(save_dir, f"model_step{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": dfm_model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "K": K,
                },
                ckpt_path,
            )
            logger.info(f"Saved: {ckpt_path}")
            
            probability_denoiser = DFMModelWrapper(dfm_model)

            if debugging is True:
                sampling_method = sampling_debugging
            else:
                sampling_method = sampling_batch

            hyps, targets, asr_hyps, str_gts = sampling_method(
                test_dl=eval_loader,
                model=probability_denoiser,
                n_step=args.n_step,
                K=K,
                max_output_length=args.max_output_length,
                mask_id=mask_id,
                return_intermediates=True,
                is_uniform=False,
                device=device,
            )

            str_hyps = []
            str_targets = []
            str_asr_hyps = []
            str_ground_truths = []
            correct_predictions = 0
            B = len(hyps)
            #TODO
            total_samples = B
            for b in range(B):
                hyp_ids = hyps[b]
                target_ids = targets[b]
                asr_hyp_ids = asr_hyps[b]
                str_gt = str_gts[b]
                
                hyp = tokenizer.decode(hyp_ids, group_tokens=False)
                target = tokenizer.decode(target_ids, group_tokens=False)
                asr_hyp = tokenizer.decode(asr_hyp_ids, group_tokens=False)
                if args.verbose:
                    logger.info(f"GT: {str_gt.split(' ')}")
                    logger.info(f"TARGET: {target.split(' ')}")
                    logger.info(f"ASR HYP: {asr_hyp.split(' ')}")
                    logger.info(f"DFM HYP: {hyp.split(' ')}")
                    logger.info("-----")            
                if hyp == target:
                    correct_predictions += 1

                str_hyps.append(hyp)
                str_targets.append(target)
                str_asr_hyps.append(asr_hyp)
                str_ground_truths.append(str_gt)

            # compute WER
            wer_score = wer(str_targets, str_hyps)
            logger.info(f"DFM WER: {wer_score * 100:.4f}%")    

            wer_score = wer(str_targets, str_asr_hyps)
            logger.info(f"ASR WER: {wer_score * 100:.4f}%")

            wer_score = wer(str_targets, str_ground_truths)
            logger.info(f"Ground Truth WER: {wer_score * 100:.4f}%")

            accuracy = correct_predictions / total_samples
            logger.info(f"Exact Matching: {accuracy * 100:.4f}% ({correct_predictions}/{total_samples})")

            total = len(hyps)
            correct = 0
            for hyp, target in zip(hyps, targets):
                if hyp == target:
                    correct += 1
            logger.info(f"Step {step} - Exact Matching: {correct/total * 100:.4f} ({correct}/{total})")

    return


if __name__ == "__main__":

    args = build_parser().parse_args()
    setup_logger(args.save_dir, log_name="train")
    logger = logging.getLogger(__name__)
    
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
    device = args.device
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
    )

    logger.info(f"* DFMConfig: ")
    logger.info(json.dumps(asdict(cfg), indent=2))
    
    dfm_model = DFMModel(cfg, device=device)
    
    if args.ckpt_path is not None:        
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        state_dict = remove_module_prefix(checkpoint["model"])
        dfm_model.load_state_dict(state_dict)        
        logger.info(f"* Loaded checkpoint from {args.ckpt_path}")

    logger.info(f"{dfm_model.device=}")
    trainable_params = 0
    for model in [dfm_model.dit]:
        tmp_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params += tmp_trainable_params
        logger.info(f"Trainable Parameters: {tmp_trainable_params:,}")
    logger.info(f"* Total Trainable Parameters: {trainable_params:,}")    

    # 멀티 GPU 지원    
    primary_gpu_id = 0
    if device == "cuda" and torch.cuda.device_count() > 1:
        gpu_ids = [int(x) for x in args.gpu.split(",")]
        num_gpus = len(gpu_ids)        
        logger.info(f"{torch.cuda.device_count()} GPUs are available")
        logger.info(f"GPU option: '{args.gpu}'")
        logger.info(f"Using {num_gpus} GPUs (IDs: {gpu_ids})")
        if num_gpus > 1:
            # 지정된 GPU들만 사용
            dfm_model = torch.nn.DataParallel(dfm_model, device_ids=gpu_ids)
            # Primary device를 첫 번째 GPU로 설정
            dfm_model = dfm_model.to(f"cuda:{gpu_ids[0]}")            

    optim = torch.optim.AdamW(
        list(dfm_model.parameters()),
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
    scaler = GradScaler(enabled=(device.startswith("cuda")))
    init_step = 0

    if args.ckpt_path is not None:
        optim.load_state_dict(checkpoint["optim"])        
        scaler.load_state_dict(checkpoint["scaler"])
        init_step = checkpoint["step"] + 1
        logger.info(f"* Loaded optimizer and scaler states from {args.ckpt_path}, "
                     f"resuming from step {init_step}")    
 
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
    logger.info(f"* 전체 train 데이터 개수: {total_train_data_count:,}")

    # TODO
    # 현재는 1024개만 추출하는 것으로    
    eval_dataset = HuBERTandDeBERTaDataset(
        task=args.eval_task,
        tokenizer=tokenizer,
        feat_dir=args.dataset_path,
        debugging=args.dataset_debugging,
        debugging_num=args.dataset_debugging_num,
        use_tar=args.use_tar,
    )
    total_eval_len = len(eval_dataset)
    indices = torch.randperm(total_eval_len)[:1024]
    small_eval_dataset = Subset(eval_dataset, indices)

    eval_sampler = BatchSampler(small_eval_dataset,
                                batch_size=args.test_batch_size,
                                shuffle=True)

    eval_dl = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        num_workers=args.num_workers,
        collate_fn=hubert_and_deberta_dataset_collate_fn,
    )

    total_eval_data_count = len(eval_dl.dataset)
    logger.info(f"* 전체 eval 데이터 개수: {total_eval_data_count:,}")

    mask_id = tokenizer.convert_tokens_to_ids(args.mask_token)
    logger.info(f"* {args.mask_token}의 ID: {mask_id}")

    train_dfm(
        args=args,
        dfm_model=dfm_model,
        train_loader=train_dl,
        eval_loader=eval_dl,
        optim=optim,
        optim_scheduler=optim_scheduler,
        mask_id=mask_id,
        init_step=init_step,
        debugging=args.debugging,        
    )