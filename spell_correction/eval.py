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


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate DFM (based on U-Net) with HuBERT + DeBERTa features")

    # ---- evaluation ----
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)    
    p.add_argument("--uniform", type=bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_step", type=int, default=5)    
    p.add_argument("--sampling_method", type=str, 
                   choices=["basic"], default="basic")
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Path to load checkpoint")        

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
    p.add_argument("--dataset_path", type=str, default="./hubert_deberta_cache_retrial")
    p.add_argument("--tokenizer_model_name", type=str, default="facebook/hubert-large-ls960-ft")
    p.add_argument("--test_task", type=str, default="test")
    p.add_argument("--mask_token", type=str, default="[MASK]")
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
    tokenizer,
    dfm_model: DFMModel,
    test_loader: DataLoader,
    mask_id: int,
    debugging: bool = False,
):
    dfm_model.eval()

    device = args.device

    total_samples = len(test_loader.dataset)
    correct_predictions = 0    

    with torch.no_grad():
        if debugging:
            sampling_method = sampling_debugging
        else:
            sampling_method = sampling_batch

        wrapper = DFMModelWrapper(dfm_model)

        hyps, targets, asr_hyps, str_gts = sampling_method(
            test_dl=test_loader,
            model=wrapper,
            n_step=args.n_step,
            K=args.vocab_size,            
            mask_id=mask_id,
            return_intermediates=True,
            is_uniform=args.uniform,
            device=device,
            verbose=args.verbose,
        )

    str_hyps = []
    str_targets = []
    str_asr_hyps = []
    str_ground_truths = []
    B = len(hyps)
    for b in range(B):
        hyp_ids = hyps[b]
        target_ids = targets[b]
        asr_hyp_ids = asr_hyps[b]
        str_gt = str_gts[b]
        
        hyp = tokenizer.decode(hyp_ids, group_tokens=False)
        target = tokenizer.decode(target_ids, group_tokens=False)
        asr_hyp = tokenizer.decode(asr_hyp_ids, group_tokens=False)
        if args.verbose:
            print(f"GT: {str_gt.split(' ')}")
            print(f"TARGET: {target.split(' ')}")
            print(f"ASR HYP: {asr_hyp.split(' ')}")
            print(f"DFM HYP: {hyp.split(' ')}")
            print("-----")            
        if hyp == target:
            correct_predictions += 1

        str_hyps.append(hyp)
        str_targets.append(target)
        str_asr_hyps.append(asr_hyp)
        str_ground_truths.append(str_gt)

    # compute WER
    wer_score = wer(str_targets, str_hyps)
    print(f"DFM WER: {wer_score * 100:.4f}%")    

    wer_score = wer(str_targets, str_asr_hyps)
    print(f"ASR WER: {wer_score * 100:.4f}%")

    wer_score = wer(str_targets, str_ground_truths)
    print(f"Ground Truth WER: {wer_score * 100:.4f}%")

    accuracy = correct_predictions / total_samples
    print(f"Exact Matching: {accuracy * 100:.4f}% ({correct_predictions}/{total_samples})")

    return accuracy


def main(args):
    
    device = args.device
    # tokenizer
    processor = AutoProcessor.from_pretrained(args.tokenizer_model_name)
    # adding numbers from 0 to 9 + "[MASK]" if not already present
    new_tokens = [str(i) for i in range(10)] + [args.mask_token]
    num_added = processor.tokenizer.add_tokens(new_tokens)
    print(f"{num_added} tokens added to the tokenizer.")
    tokenizer = processor.tokenizer

    # num tokens 확인
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != args.vocab_size:
        print(f"Warning: vocab_size argument ({args.vocab_size}) "
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
        embed_dim=args.embed_dim,
        length_hidden_dim=args.length_hidden_dim,
        max_target_positions=args.max_target_positions,
        length_dropout=args.length_dropout,
    )        
    print(f"* DFMConfig: ")
    print(json.dumps(asdict(cfg), indent=2))
    
    dfm_model = DFMModel(cfg, device=device)

    assert os.path.exists(args.ckpt_path), f"Checkpoint path {args.ckpt_path} does not exist."    
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    dfm_model.load_state_dict(checkpoint["dfm_model"])
        
    print(f"* Loaded checkpoint from {args.ckpt_path}")

    test_dataset = HuBERTandDeBERTaDataset(
        task=args.test_task,
        tokenizer=tokenizer,
        feat_dir=args.dataset_path,
        debugging=args.debugging,
        debugging_num=args.debugging_num,
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

    device = args.device

    MASK = args.mask_token
    mask_id = tokenizer.convert_tokens_to_ids(MASK)

    eval_dfm(
        args=args,
        tokenizer=tokenizer,
        dfm_model=dfm_model.to(device),
        test_loader=test_dl,
        mask_id=mask_id,
        debugging=args.debugging,
    )

    return

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))

    set_seed(args.seed)

    main(args)