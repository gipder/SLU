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
from pathlib import Path
from typing import List

from transformers import AutoProcessor
import os
from jiwer import wer, process_words, process_characters

# my implementation
from model import ARModel, ARModelConfig
from hubert_deberta_dataset import HuBERTandDeBERTaDataset
from hubert_deberta_dataset import hubert_and_deberta_dataset_collate_fn
from hubert_deberta_dataset import BatchSampler

# my implementation
from utils import compute_wer_cer, compute_metrics
from utils import set_seed, seed_worker
from utils import setup_logger
from utils import str2bool, class_name
from utils import replace_digit_in_spoken_text
from tqdm import tqdm



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


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate DiT with HuBERT + DeBERTa features")

    # ---- evaluation ----
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_step", type=int, default=100, help="Logging step interval during training")
    p.add_argument("--num_workers", type=int, default=4)        
    p.add_argument("--seed", type=int, default=42)    
    p.add_argument("--gpu", type=str, default="0",
                   help="GPU ids to use, e.g., '0,1,2'") 
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Path to load checkpoint")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Directory to save evaluation logs")
    p.add_argument("--use_cache", type=str2bool, default=True, help="Whether to use cache during evaluation")
    p.add_argument("--avg", type=int, default=1,
                   help="Number of checkpoints to average. If >= 2, averages from epoch-avg+1 to epoch.")

    # ---- model dims / arch ----
    ## for DiT model
    p.add_argument("--vocab_size", type=int, default=47)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--audio_dim", type=int, default=1024)
    p.add_argument("--text_dim", type=int, default=1024)
    p.add_argument("--max_output_length", type=int, default=512)
    p.add_argument("--model_type", type=str, choices=["transformer", "encoder_decoder_transformer", "fused_transformer"], default="transformer")
    p.add_argument("--norm_first", type=str2bool, default=True, help="Whether to apply layer normalization before attention and FFN")    

    ## for length predictor
    #p.add_argument("--embed_dim", type=int, default=1024)
    #p.add_argument("--length_hidden_dim", type=int, default=512)        
    #p.add_argument("--length_condition", type=str, choices=["audio", "text", "both"], default="text")
    #p.add_argument("--length_margin", type=float, default=0.1)

    p.add_argument("--use_intent_token", type=str2bool, default=False,
                   help="Whether to add INTENT-derived tokens (and sentencepiece variants) to tokenizer")
    p.add_argument("--prompt_tag_path", type=str, default="data/slu/INTENT",
                   help="Path to INTENT label file used for intent token injection")

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


def _average_checkpoints(ckpt_path: str, avg: int, device: torch.device):
    """Average `avg` consecutive checkpoints ending at `ckpt_path`.

    Returns (averaged_state_dict, list_of_averaged_paths).
    Skips missing checkpoints with a warning.
    """
    import re as _re
    path = Path(ckpt_path)
    m = _re.search(r"epoch(\d+)", path.name)
    if m is None:
        raise ValueError(
            f"Cannot infer epoch number from checkpoint name '{path.name}'. "
            "Expected pattern: model_epoch<N>.pt"
        )
    end_epoch = int(m.group(1))
    start_epoch = end_epoch - avg + 1

    averaged_paths = []
    state_dicts = []
    for ep in range(start_epoch, end_epoch + 1):
        candidate = path.parent / path.name.replace(f"epoch{end_epoch}", f"epoch{ep}")
        if not candidate.exists():
            logger.warning(f"Checkpoint not found, skipping: {candidate}")
            continue
        ckpt = torch.load(candidate, map_location=device)
        state_dicts.append(ckpt["model"])
        averaged_paths.append(str(candidate))
        logger.info(f"  averaging: {candidate.name}")

    if not state_dicts:
        raise FileNotFoundError(f"No checkpoints found in range epoch {start_epoch}–{end_epoch}.")

    # Average all state dicts
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = sum(sd[key].float() for sd in state_dicts) / len(state_dicts)
        avg_state[key] = avg_state[key].to(state_dicts[0][key].dtype)

    return avg_state, averaged_paths


def eval_model(
    args: ArgumentParser,
    task: str,
    tokenizer,
    model: ARModel,
    test_loader: DataLoader,
    sos_id: int = 1,
    eos_id: int = 2,
):
    model.eval()
    device = (
        next(model.parameters()).device
        if not isinstance(model, torch.nn.DataParallel) 
        else next(model.module.parameters()).device
    )    

    total_samples = len(test_loader.dataset)    

    with torch.no_grad():
        hyp_ids = []
        target_ids = []
        str_asr_hyps = []
        str_asr_gts = []
        str_slus = []
        step = 0
        count = 0
        for batch in tqdm(test_loader):
            audio_feats = batch["feat"]
            audio_feat_mask = batch["feat_mask"]
            text_feats = batch["text_feat"]
            text_feat_mask = batch["text_mask"]
            slus = batch["slu"]
            slu_mask = batch["slu_mask"]
            str_asr_hypothesis = batch["str_hyp"]
            str_asr_gt = batch["str_gt"]
            str_slu_targets = batch["str_slu"]

            audio_feats = audio_feats.to(device) # B, T, D
            audio_feat_mask = audio_feat_mask.to(device)
            text_feats = text_feats.to(device)
            text_feat_mask = text_feat_mask.to(device)
            slus = slus.to(device)
            slu_mask = slu_mask.to(device)

            generated = model.decode(
                audio_feats=audio_feats,
                text_feats=text_feats,
                audio_mask=audio_feat_mask,
                text_mask=text_feat_mask,
                max_output_length=args.max_output_length,
                sos_id=sos_id,
                eos_id=eos_id,  
                use_cache=args.use_cache,              
                device=device,
            )

            hyp_ids.extend(generated.cpu().tolist())
            target_ids.extend(slus.cpu().tolist())
            str_asr_hyps.extend(str_asr_hypothesis)
            str_asr_gts.extend(str_asr_gt)
            str_slus.extend(str_slu_targets)

            step += 1
            count += audio_feats.size(0)

            if args.verbose:
                logger.info(f"{slus=}")
                logger.info(f"{generated=}")
                logger.info(f"{slus.shape=}")
                logger.info(f"{generated.shape=}")
            if step % args.log_step == 0:
                logger.info(f"Evaluation step {step:,}/{len(test_loader):,} completed.")
                logger.info(f"  Processed {count:,}/{total_samples:,} samples.")            
            
            if args.debugging and count >= args.debugging_num:
                logger.info(f"Debugging mode: Stopping evaluation after {count} samples.")
                break

    # id to string conversion
    blank_id = tokenizer.pad_token_id    
    str_hyps = []
    str_targets = []
    correct_predictions = 0

    if args.debugging is True:
        logger.info(f"Total samples in debugging mode: {len(hyp_ids)}")
        total_samples = args.debugging_num

    assert total_samples == len(hyp_ids) == len(target_ids), \
        f"Number of hypotheses ({len(hyp_ids)}) does not match number of targets ({len(target_ids)})"
    for b in range(len(hyp_ids)):
        hyp_id = hyp_ids[b]
        target_id = target_ids[b]
        #str_asr_hyp = str_asr_hyps[b]
        #str_gt = str_slus
        #str_sc_target = str_sc_targets[b]
        # Remove blank tokens
        hyp_ids_cleaned = [id for id in hyp_id if id != blank_id]
        target_ids_cleaned = [id for id in target_id if id != blank_id]
        
        hyp = tokenizer.decode(hyp_ids_cleaned, group_tokens=False, skip_special_tokens=True)
        target = tokenizer.decode(target_ids_cleaned, group_tokens=False, skip_special_tokens=True)
        if args.verbose:
            logger.info(f"Hypothesis: {hyp}")
            logger.info(f"Reference:  {target}")
            logger.info("-----")

        if hyp == target:
            correct_predictions += 1
        
        str_hyps.append(hyp)
        str_targets.append(target)
    results = compute_metrics(str_hyps, str_targets)
    str_asr_hyps_norm = [replace_digit_in_spoken_text(t).upper() for t in str_asr_hyps]
    str_asr_gts_norm  = [replace_digit_in_spoken_text(t).upper() for t in str_asr_gts]
    asr_results = compute_wer_cer(str_asr_hyps_norm, str_asr_gts_norm)        
    gt_results = compute_wer_cer(str_targets, str_slus)

    # compute WER
    #result = {}
    #wer_score = wer(str_targets, str_hyps_decoded)
    logger.info(f"SLU WER: {results['wer'] * 100:.4f}%")
    logger.info(f"SLU SER: {results['ser'] * 100:.4f}%")
    logger.info(f"SLU EM: {results['em'] * 100:.4f}%")
    #logger.info(f"SLU EM Tree: {results['em_tree'] * 100:.4f}%")
    #result["dfm_wer"] = wer_score

    #wer_score = wer(str_targets, str_hyps)
    results["asr_wer"] = asr_results["wer"]
    logger.info(f"ASR WER: {results['asr_wer'] * 100:.4f}%")

    results["gt_wer"] = gt_results["wer"]
    logger.info(f"GT WER: {results['gt_wer'] * 100:.4f}%")
    #result["asr_wer"] = wer_score

    # adding senteces for debugging
    results["sentences"] = []
    for i in range(len(str_hyps)):
        results["sentences"].append({
            "ground_truth": str_targets[i],
            "slu_hypothesis": str_hyps[i],
            "asr_hypothesis": str_asr_hyps[i],            
        })
        
    return results


def main(args):
 
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

    logger.info(f"* Model Config: {class_name(cfg)}")
    logger.info(json.dumps(asdict(cfg), indent=2))
    
    model = ARModel(cfg)

    assert os.path.exists(args.ckpt_path), f"Checkpoint path {args.ckpt_path} does not exist."

    # if args.ckpt_path is a link, resolve the link to get the actual checkpoint path
    if os.path.islink(args.ckpt_path):
        ckpt_dir = os.path.dirname(args.ckpt_path)
        resolved_path = os.readlink(args.ckpt_path)
        logger.info(f"Resolved checkpoint path: {resolved_path} "
                    f"from symbolic link: {args.ckpt_path}")
        args.ckpt_path = resolved_path
        if not os.path.exists(args.ckpt_path):
            basename = os.path.basename(args.ckpt_path)
            args.ckpt_path = os.path.join(ckpt_dir, basename)
            logger.info(f"Resolved checkpoint path does not exist. "
                        f"Converted to path: {args.ckpt_path}")

    averaged_ckpt_paths = [args.ckpt_path]
    if args.avg >= 2:
        logger.info(f"* Checkpoint averaging: avg={args.avg}")
        avg_state, averaged_ckpt_paths = _average_checkpoints(args.ckpt_path, args.avg, device)
        model.load_state_dict(avg_state)
        logger.info(f"* Averaged {len(averaged_ckpt_paths)} checkpoints: {[os.path.basename(p) for p in averaged_ckpt_paths]}")
    else:
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        logger.info(f"* Loaded checkpoint from {args.ckpt_path}")

    trainable_params = 0
    for model_part in [model.slu_model]:
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
    sos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    logger.info(f"Mask token: '{MASK}' (ID: {mask_id})")
    logger.info(f"SOS token ID: {sos_id}")
    logger.info(f"EOS token ID: {eos_id}")

    slu_wers = []    
    slu_ems = []
    #slu_em_trees = []
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
            sos_id=sos_id,
            eos_id=eos_id,
        )
        results["ckpt"] = args.ckpt_path
        results["avg"] = args.avg
        results["averaged_ckpts"] = averaged_ckpt_paths
        slu_wers.append(results["wer"])
        slu_ems.append(results["em"])
        #slu_em_trees.append(results["em_tree"])
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
        if args.avg >= 2 and len(averaged_ckpt_paths) >= 2:
            import re as _re
            epochs = []
            for p in averaged_ckpt_paths:
                m = _re.search(r"epoch(\d+)", os.path.basename(p))
                if m:
                    epochs.append(m.group(1))
            if epochs:
                model_name = f"averaged_{epochs[0]}-{epochs[-1]}_" + model_name
        results_path = os.path.join(
            args.save_dir,
            f"{task}_{model_name}_{time_id}_eval_summary.json",
        )

        # Separate sentences from results
        sentences = results.pop("sentences", [])

        # Save main results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved evaluation results to {results_path}")

        # Save sentences to separate file
        sentences_path = os.path.join(
            args.save_dir,
            f"{task}_{model_name}_{time_id}_eval_sentences.json",
        )
        with open(sentences_path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, indent=4)
        logger.info(f"Saved sentences to {sentences_path}")    

    logger.info("=" * 60)
    logger.info(f"{os.path.basename(args.ckpt_path)} EVALUATION SUMMARY")
    if args.avg >= 2:
        logger.info(f"Checkpoint averaging: {len(averaged_ckpt_paths)} ckpts averaged "
                    f"({[os.path.basename(p) for p in averaged_ckpt_paths]})")
    logger.info("-" * 60)
    for i, task in enumerate(args.test_task):    
        logger.info(f"Total samples in {task} set: {results['num_sentences']}")    
        logger.info(f"{task} SLU WER: {slu_wers[i] * 100:.4f}%")        
        logger.info(f"{task} SLU EM: {slu_ems[i] * 100:.4f}%")
        #logger.info(f"{task} SLU EM Tree: {slu_em_trees[i] * 100:.4f}%")
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
    logger.info(json.dumps(vars(args), indent=2))

    set_seed(args.seed)    

    main(args)
