import logging
import sys
import torch
from torch.utils.data import DataLoader

import argparse
from argparse import ArgumentParser
import json
from dataclasses import asdict

from transformers import AutoProcessor
import os
from jiwer import wer

# my implementation
from model import DFMModel, DFMModelConfig, DFMModelWrapper
from hubert_deberta_dataset import HuBERTandDeBERTaDataset
from hubert_deberta_dataset import hubert_and_deberta_dataset_collate_fn
from hubert_deberta_dataset import BatchSampler
from sampling import sampling_batch, sampling_debugging
from utils import set_seed, seed_worker
from utils import setup_logger
from utils import str2bool


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate DiT with HuBERT + DeBERTa features")

    # ---- evaluation ----
    p.add_argument("--batch_size", type=int, default=256)
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
    ## for length predictor
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--length_hidden_dim", type=int, default=512)
    p.add_argument("--max_target_positions", type=int, default=256)
    p.add_argument("--length_dropout", type=float, default=0.1)
    p.add_argument("--length_condition", type=str, choices=["audio", "text"], default="text")

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
    p.add_argument("--debugging", action="store_true", default=False, help="Enable debugging mode for train_dfm()")    
    p.add_argument("--debugging_num", type=int, default=128, help="How many samples are used in debugging")
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--use_oracle_length", action="store_true", default=False)
    
    # ---- device ----
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    return p


def eval_dfm(
    args: ArgumentParser,
    task: str,
    tokenizer,
    dfm_model: DFMModel,
    test_loader: DataLoader,
    mask_id: int,    
):
    dfm_model.eval()

    device = dfm_model.device \
        if isinstance(dfm_model, DFMModel) \
        else dfm_model.module.device

    total_samples = len(test_loader.dataset)
    correct_predictions = 0    

    with torch.no_grad():
        if args.debugging:
            sampling_method = sampling_debugging
        else:
            sampling_method = sampling_batch

        wrapper = DFMModelWrapper(dfm_model)

        hyps, targets, str_hyps, str_gts = sampling_method(
            test_dl=test_loader,
            model=wrapper,
            n_step=args.n_step,
            K=args.vocab_size,            
            mask_id=mask_id,
            return_intermediates=True,
            is_uniform=args.uniform,
            device=device,
            verbose=args.verbose,
            debugging=args.debugging,
        )

    str_hyps_decoded = []
    str_targets = []
    str_ground_truths = str_gts  # 이미 문자열 형태
    
    B = len(hyps)
    for b in range(B):
        hyp_ids = hyps[b]
        target_ids = targets[b]
        
        hyp = tokenizer.decode(hyp_ids, group_tokens=False)
        target = tokenizer.decode(target_ids, group_tokens=False)
        
        if args.verbose:
            logger.info(f"GT: {str_gts[b].split(' ')}")
            logger.info(f"TARGET: {target.split(' ')}")
            logger.info(f"ASR HYP: {str_hyps[b].split(' ')}")
            logger.info(f"DFM HYP: {hyp.split(' ')}")
            logger.info("-----")            
        if hyp == target:
            correct_predictions += 1

        str_hyps_decoded.append(hyp)
        str_targets.append(target)

    # compute WER
    result = {}
    wer_score = wer(str_targets, str_hyps_decoded)
    logger.info(f"DFM WER: {wer_score * 100:.4f}%")    
    result["dfm_wer"] = wer_score

    wer_score = wer(str_targets, str_hyps)
    logger.info(f"ASR WER: {wer_score * 100:.4f}%")
    result["asr_wer"] = wer_score

    wer_score = wer(str_targets, str_ground_truths)
    logger.info(f"Ground Truth WER: {wer_score * 100:.4f}%")
    result["gt_wer"] = wer_score

    accuracy = correct_predictions / total_samples
    logger.info(f"Exact Matching: {accuracy * 100:.4f}% ({correct_predictions}/{total_samples})")    
    result["accuracy"] = accuracy

    return result


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
        max_output_length=args.max_target_positions,   
        n_step=args.n_step,     
    )        
    logger.info(f"* DFMConfig: ")
    logger.info(json.dumps(asdict(cfg), indent=2))
    
    dfm_model = DFMModel(cfg, device=device)

    assert os.path.exists(args.ckpt_path), f"Checkpoint path {args.ckpt_path} does not exist."    
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    dfm_model.load_state_dict(checkpoint["model"])
        
    logger.info(f"* Loaded checkpoint from {args.ckpt_path}")

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
            dfm_model = dfm_model.to(device)                        
            dfm_model = torch.nn.DataParallel(dfm_model,
                                              device_ids=gpu_ids,
                                              output_device=primary_gpu_id)
        else:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            dfm_model = dfm_model.to(device)
    else:
        logger.info("Using CPU device")
        dfm_model = dfm_model.to(device)          

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

        results = eval_dfm(
            args=args,
            task=task,
            tokenizer=tokenizer,
            dfm_model=dfm_model,
            test_loader=test_dl,
            mask_id=mask_id,            
        )
        dfm_wers.append(results["dfm_wer"])
        asr_wers.append(results["asr_wer"])
        gt_wers.append(results["gt_wer"])

    logger.info("=" * 60)
    logger.info(f"{os.path.basename(args.ckpt_path)} EVALUATION SUMMARY")
    logger.info("-" * 60)
    for i, task in enumerate(args.test_task):        
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

    logger = logging.getLogger("__name__")
        
    logger.info("Command line: " + " ".join(sys.argv))
    logger.info(f"Evaluation logs will be saved to {args.save_dir}" )
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=2))              

    set_seed(args.seed)    

    main(args)