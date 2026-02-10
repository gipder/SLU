import logging
import sys
from tabnanny import verbose
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from argparse import ArgumentParser
import json
from dataclasses import asdict

from transformers import AutoProcessor
import os
from jiwer import wer, process_words, process_characters

# my implementation
from model import ARModel, ARModelConfig
from hubert_deberta_dataset import HuBERTandDeBERTaDataset
from hubert_deberta_dataset import hubert_and_deberta_dataset_collate_fn
from hubert_deberta_dataset import BatchSampler
from utils import compute_wer_cer
from utils import set_seed, seed_worker
from utils import setup_logger
from utils import str2bool


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate DiT with HuBERT + DeBERTa features")

    # ---- evaluation ----
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=str, default="0",
                   help="GPU ids to use, e.g., '0,1,2'")
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
    p.add_argument("--max_output_length", type=int, default=256, help="Maximum output length during inference")

    # ---- data / tokenization ----
    p.add_argument("--dataset_path", type=str, default="./hubert_deberta_tar")
    p.add_argument("--tokenizer_model_name", type=str, default="facebook/hubert-large-ls960-ft")
    p.add_argument("--test_task", nargs="+", type=str, default=["test-clean","test-other"],
                   help="Testing task name (default: test-clean,test-other)")
    p.add_argument("--mask_token", type=str, default="[MASK]")
    p.add_argument("--use_tar", type=str2bool, default=True,
                   help="Whether to use .tar files for dataset")

    # ---- debugging ----
    p.add_argument("--debugging", type=str2bool, default=False, help="Enable debugging mode for eval_dfm()")
    p.add_argument("--debugging_num", type=int, default=128, help="How many samples are used in debugging")
    p.add_argument("--verbose", type=str2bool, default=False)    

    # ---- device ----
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    return p


def eval_model(
    args: ArgumentParser,
    task: str,
    tokenizer,
    model: ARModel,
    test_loader: DataLoader,
    sos_id: int,
    eos_id: int,
):
    model.eval()

    device = (
        next(model.parameters()).device
        if not isinstance(model, torch.nn.DataParallel)
        else next(model.module.parameters()).device
    )

    total_samples = len(test_loader.dataset)
    correct_predictions = 0

    with torch.no_grad():
        id_hyps = []
        id_targets = []
        str_asr_hyps_from_batch = []
        str_gts_from_batch = []
        for batch in tqdm(test_loader):
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

            id_hyps.extend(generated.cpu().tolist())
            id_targets.extend(gts.cpu().tolist())
            #str_hyps is tuple 
            for j in range(len(str_hyps)):
                str_asr_hyps_from_batch.append(str_hyps[j])
                str_gts_from_batch.append(str_gts[j])
            if args.debugging:
                if len(id_hyps) >= args.debugging_num:
                    break

    # id to string conversion
    blank_id = tokenizer.pad_token_id
    hyp_texts = []
    ref_texts = []
    for hyp_ids, target_ids in zip(id_hyps, id_targets):
        # Remove blank tokens
        hyp_ids_cleaned = [id for id in hyp_ids if id != blank_id]
        target_ids_cleaned = [id for id in target_ids if id != blank_id]

        hyp_text = tokenizer.decode(hyp_ids_cleaned, group_tokens=False, skip_special_tokens=True)
        ref_text = tokenizer.decode(target_ids_cleaned, group_tokens=False, skip_special_tokens=True)
        if args.verbose:
            logger.info(f"Hypothesis: {hyp_text}")
            logger.info(f"Reference:  {ref_text}")
            logger.info("-----")

        hyp_texts.append(hyp_text)
        ref_texts.append(ref_text)

    results = compute_wer_cer(hyp_texts, ref_texts)
    asr_results = compute_wer_cer(str_asr_hyps_from_batch, str_gts_from_batch)
    gt_results = compute_wer_cer(ref_texts, str_gts_from_batch)

    # compute WER
    #result = {}
    #wer_score = wer(str_targets, str_hyps_decoded)
    logger.info(f"Correction1 WER: {results['wer'] * 100:.4f}%")
    #result["dfm_wer"] = wer_score

    #wer_score = wer(str_targets, str_hyps)
    results["asr_wer"] = asr_results["wer"]
    logger.info(f"ASR WER: {results['asr_wer'] * 100:.4f}%")

    results["gt_wer"] = gt_results["wer"]
    logger.info(f"GT WER: {results['gt_wer'] * 100:.4f}%")
    #result["asr_wer"] = wer_score

    # adding senteces for debugging
    results["sentences"] = []
    for i in range(len(str_gts_from_batch)):
        results["sentences"].append({
            "ground_truth": str_gts_from_batch[i],
            "asr_hypothesis": str_asr_hyps_from_batch[i],
            "correction_hypothesis": hyp_texts[i],
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

    cfg = ARModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        audio_dim=args.audio_dim,
        text_dim=args.text_dim,
        max_output_length=args.max_output_length,        
    )
    logger.info(f"* ARModelConfig: ")
    logger.info(json.dumps(asdict(cfg), indent=2))

    ar_model = ARModel(cfg)

    assert os.path.exists(args.ckpt_path), f"Checkpoint path {args.ckpt_path} does not exist."

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    ar_model.load_state_dict(checkpoint["model"])

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
            ar_model = ar_model.to(device)
            ar_model = torch.nn.DataParallel(ar_model,
                                              device_ids=gpu_ids,
                                              output_device=primary_gpu_id)
        else:
            device = torch.device(f"cuda:{gpu_ids[0]}")
            ar_model = ar_model.to(device)
            logger.info(f"Using single GPU: cuda:{gpu_ids[0]}")
    else:
        if device.type == "cuda":
            gpu_ids = [int(x) for x in args.gpu.split(",")]
            device = torch.device(f"cuda:{gpu_ids[0]}")
            ar_model = ar_model.to(device)
            logger.info(f"Using single GPU: {device}")
        else:
            ar_model = ar_model.to(device)
            logger.info("Using CPU device")

    MASK = args.mask_token
    mask_id = tokenizer.convert_tokens_to_ids(MASK)
    sos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    logger.info(f"Start token: '{tokenizer.bos_token}' (ID: {sos_id})")
    logger.info(f"End token: '{tokenizer.eos_token}' (ID: {eos_id})")
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
            model=ar_model,
            test_loader=test_dl,
            sos_id=sos_id,
            eos_id=eos_id,
        )
        dfm_wers.append(results["wer"])
        asr_wers.append(results["asr_wer"])
        gt_wers.append(results["gt_wer"])

        #json dump in save_dir
        model_name = os.path.basename(args.ckpt_path).replace(".pt", "").replace(".pth", "")
        results_path = os.path.join(args.save_dir, f"{task}_{model_name}_eval_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved evaluation results to {results_path}")

    logger.info("=" * 60)
    logger.info(f"{os.path.basename(args.ckpt_path)} EVALUATION SUMMARY")
    logger.info("-" * 60)
    for i, task in enumerate(args.test_task):
        logger.info(f"Total samples in {task} set: {results['num_sentences']}")
        logger.info(f"{task} Correction WER: {dfm_wers[i] * 100:.4f}%")
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