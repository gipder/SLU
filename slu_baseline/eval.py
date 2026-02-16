import logging
import re
import sys
from tabnanny import verbose
import torch
from torch.utils.data import DataLoader

import argparse
from argparse import ArgumentParser
import json
from dataclasses import asdict

from transformers import AutoProcessor
import os
from jiwer import wer, process_words, process_characters

# my implementation
from model import DFMModel, DFMModelConfig, DFMModelWrapper
from hubert_deberta_dataset import HuBERTandDeBERTaDataset
from hubert_deberta_dataset import hubert_and_deberta_dataset_collate_fn
from hubert_deberta_dataset import BatchSampler
from sampling import sampling_batch, sampling_debugging
from utils import compute_wer_cer
from utils import set_seed, seed_worker
from utils import setup_logger
from utils import str2bool
from tqdm import tqdm

# from flow_matching
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver
# my implementation
from custom_path import UniformDiscreteProbPath
from utils import compute_wer_cer

def class_name(obj) -> str:
    return type(obj).__name__


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate DiT with HuBERT + DeBERTa features")

    # ---- evaluation ----
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_step", type=int, default=10, help="Logging step interval during training")
    p.add_argument("--num_workers", type=int, default=4)    
    p.add_argument("--uniform", type=bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_step", type=int, default=5)   
    p.add_argument("--gpu", type=str, default="0",
                   help="GPU ids to use, e.g., '0,1,2'") 
    p.add_argument("--sampling_method", type=str, 
                   choices=["basic"], default="basic")
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Path to load checkpoint")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Directory to save evaluation logs")

    # ---- model dims / arch ----
    ## for DiT model
    p.add_argument("--vocab_size", type=int, default=43)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--audio_dim", type=int, default=1024)
    p.add_argument("--text_dim", type=int, default=1024)
    p.add_argument("--max_output_length", type=int, default=256)
    p.add_argument("--noise_ratio", type=float, default=0.5, help="Noise ratio for UniformDiscreteProbPath")
    p.add_argument("--model_type", type=str, choices=["dit", "transformer"], default="dit")

    ## for length predictor
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--length_hidden_dim", type=int, default=512)        
    p.add_argument("--length_condition", type=str, choices=["audio", "text", "both"], default="text")
    p.add_argument("--length_margin", type=float, default=0.1)

    # ---- data / tokenization ----
    p.add_argument("--dataset_path", type=str, default="./hubert_deberta_tar")
    p.add_argument("--tokenizer_model_name", type=str, default="facebook/hubert-large-ls960-ft")
    p.add_argument("--test_task", nargs="+", type=str, default=["test-clean","test-other"],
                   help="Testing task name (default: test-clean,test-other)")    
    p.add_argument("--mask_token", type=str, default="[MASK]")
    p.add_argument("--use_tar", type=str2bool, default=True,
                   help="Whether to use .tar files for dataset")
    #p.add_argument("--shuffle_train", type=bool, default=True)

    # ---- debugging ----
    p.add_argument("--debugging", type=str2bool, default=False, help="Enable debugging mode for eval_dfm()")    
    p.add_argument("--debugging_num", type=int, default=128, help="How many samples are used in debugging")
    p.add_argument("--verbose", type=str2bool, default=False)
    p.add_argument("--use_oracle_length", type=str2bool, default=False)
    
    # ---- device ----
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    return p


def eval_model(
    args: ArgumentParser,
    task: str,
    tokenizer,
    model: DFMModel,
    test_loader: DataLoader,
    mask_id: int,    
):
    model.eval()
    device = (
        next(model.parameters()).device
        if not isinstance(model, torch.nn.DataParallel) 
        else next(model.module.parameters()).device
    )    

    total_samples = len(test_loader.dataset)    

    with torch.no_grad():
        wrapper = DFMModelWrapper(model)
        scheduler = PolynomialConvexScheduler(n=2.0)
        path = UniformDiscreteProbPath(
            scheduler=scheduler,
            vocab_size=args.vocab_size,
            noise_ratio=args.noise_ratio,
        )

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
        for batch in tqdm(test_loader):
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

            if args.debugging:
                logger.info(f"{gts=}")
                logger.info(f"{x_1_hat[-1]=}")
                logger.info(f"{gts.shape=}")
                logger.info(f"{x_1_hat[-1].shape=}")
            if step % args.log_step == 0:
                logger.info(f"Evaluation step {step:,}/{len(test_loader):,} completed.")
                logger.info(f"  Processed {count:,}/{total_samples:,} samples.")            

            

    # id to string conversion
    blank_id = tokenizer.pad_token_id    
    hyp_texts = []
    ref_texts = []
    correct_predictions = 0
    assert len(hyp_ids) == len(target_ids), \
        f"Number of hypotheses ({len(hyp_ids)}) does not match number of targets ({len(target_ids)})"
    for b in range(len(hyp_ids)):
        hyp_id = hyp_ids[b]
        target_id = target_ids[b]
        #str_asr_hyp = str_asr_hyps[b]
        #str_sc_target = str_sc_targets[b]
        # Remove blank tokens
        hyp_ids_cleaned = [id for id in hyp_id if id != blank_id]
        target_ids_cleaned = [id for id in target_id if id != blank_id]
        
        hyp_text = tokenizer.decode(hyp_ids_cleaned, group_tokens=False, skip_special_tokens=True)
        ref_text = tokenizer.decode(target_ids_cleaned, group_tokens=False, skip_special_tokens=True)
        if args.verbose:
            logger.info(f"Hypothesis: {hyp_text}")
            logger.info(f"Reference:  {ref_text}")
            logger.info("-----")
        
        hyp_texts.append(hyp_text)
        ref_texts.append(ref_text)

    results = compute_wer_cer(hyp_texts, ref_texts)
    asr_results = compute_wer_cer(str_asr_hyps, str_sc_targets)        
    gt_results = compute_wer_cer(ref_texts, str_sc_targets)

    # compute WER
    #result = {}
    #wer_score = wer(str_targets, str_hyps_decoded)
    logger.info(f"DFM WER: {results['wer'] * 100:.4f}%")    
    #result["dfm_wer"] = wer_score

    #wer_score = wer(str_targets, str_hyps)
    results["asr_wer"] = asr_results["wer"]
    logger.info(f"ASR WER: {results['asr_wer'] * 100:.4f}%")

    results["gt_wer"] = gt_results["wer"]
    logger.info(f"GT WER: {results['gt_wer'] * 100:.4f}%")
    #result["asr_wer"] = wer_score

    # adding senteces for debugging
    results["sentences"] = []
    for i in range(len(ref_texts)):
        results["sentences"].append({
            "ground_truth": ref_texts[i],
            "asr_hypothesis": str_asr_hyps[i],
            "dfm_hypothesis": hyp_texts[i],
        })
        
    return results


def main(args):
 
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
    assert actual_vocab_size == args.vocab_size, \
        f"Tokenizer vocab size ({actual_vocab_size}) does not match args.vocab_size ({args.vocab_size})"

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
    )        

    logger.info(f"* Model Config: {class_name(cfg)}")
    logger.info(json.dumps(asdict(cfg), indent=2))
    
    model = DFMModel(cfg)

    assert os.path.exists(args.ckpt_path), f"Checkpoint path {args.ckpt_path} does not exist."    
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
        
    logger.info(f"* Loaded checkpoint from {args.ckpt_path}")

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

    MASK = args.mask_token
    mask_id = tokenizer.convert_tokens_to_ids(MASK)
    logger.info(f"Mask token: '{MASK}' (ID: {mask_id})")

    dfm_wers = []    
    asr_wers = []
    gt_wers = []
    for task in args.test_task:
        logger.info(f"===== Evaluating on {task} set =====")        
        test_dataset = HuBERTandDeBERTaDataset(
            task=task,
            tokenizer=tokenizer,
            feat_dir=args.dataset_path,
            debugging=args.debugging,
            debugging_num=args.debugging_num,
            use_tar=args.use_tar,
        )

        test_sampler = BatchSampler(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)

        test_dl = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=hubert_and_deberta_dataset_collate_fn,
        )

        results = eval_model(
            args=args,
            task=task,
            tokenizer=tokenizer,
            model=model,
            test_loader=test_dl,
            mask_id=mask_id,            
        )
        dfm_wers.append(results["wer"])
        asr_wers.append(results["asr_wer"])
        gt_wers.append(results["gt_wer"])        

        #json dump in save_dir
        #unique time id from the log file
        time_id = "unknown"
        # logging file name
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_basename = os.path.basename(handler.baseFilename)
                break
        # remove suffix 
        parts = os.path.splitext(log_file_basename)[0].split("-")
        if len(parts) >= 3:
            date_part, time_part = parts[-2], parts[-1]
            if date_part.isdigit() and time_part.isdigit():
                time_id = f"{date_part}-{time_part}"
        model_name = os.path.basename(args.ckpt_path).replace(".pt", "").replace(".pth", "")
        results_path = os.path.join(
            args.save_dir,
            f"{task}_{model_name}_n-step{args.n_step}_{time_id}_eval_results.json",
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved evaluation results to {results_path}")    

    logger.info("=" * 60)
    logger.info(f"{os.path.basename(args.ckpt_path)} EVALUATION SUMMARY")
    logger.info("-" * 60)
    for i, task in enumerate(args.test_task):    
        logger.info(f"Total samples in {task} set: {results['num_sentences']}")    
        logger.info(f"{task} DFM WER: {dfm_wers[i] * 100:.4f}%")
        logger.info(f"{task} ASR WER: {asr_wers[i] * 100:.4f}%")        
        logger.info(f"{task} GT WER: {gt_wers[i] * 100:.4f}%")
        logger.info("-" * 60)    

    return

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()   

    if args.save_dir is None:
        # ckpt_path 폴더 기준으로 로그 저장 폴더 생성
        args.save_dir = os.path.join(
            os.path.dirname(args.ckpt_path),            
        )
    setup_logger(save_dir=args.save_dir, log_name="eval")

    logger = logging.getLogger()  # root logger
        
    logger.info("Command line: " + " ".join(sys.argv))
    logger.info(f"Evaluation logs will be saved to {args.save_dir}" )
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=2))              z

    set_seed(args.seed)    

    main(args)