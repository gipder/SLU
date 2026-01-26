import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from deberta_and_dit_dataset import DeBERTaAndDiTDataset, DeBERTaAndDiTCollator

import os

from utils import setup_logger, get_config, get_test_config

# For DFM sampmling
from sampling import sample

# Config
from slu_model import SLUConfig, SLUModel
from upsampler import LearnedUpsample
from projector import CTCProjector

def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def main():
    config = get_config()
    config = get_test_config(config)
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = setup_logger(save_dir)
    logger.info(f"configuration information: {config}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = config.eval.ckpt
    logger.info(f"Checkpoint Path: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    logger.info(f"Loading {checkpoint_path} completed")
    # 굳이 없어도 됨, 나중에 고민해볼 것
    model_config = SLUConfig(**checkpoint['config']['model'])

    upsample = LearnedUpsample(
        d_model=model_config.d_model,
        upsample_factor=model_config.upsample_factor,
    ).to(device)

    proj = CTCProjector(
        input_dim=model_config.d_model,
        d_model=model_config.d_model,
        vocab_size=model_config.vocab_size,
    ).to(device)

    slu_model = SLUModel(
        config=config,
        upsampler=upsample,
        proj=proj,
    ).to(device)

    slu_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    slu_model.eval()

    # output tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(config.data.bpe_model_path)
    logger.info(f"Loading {config.data.bpe_model_path} completed")
    tokenizer = AutoTokenizer.from_pretrained(config.model.llm_name)

    blank = config.data.blank
    MASK = config.data.MASK
    in_tokenizer = tokenizer
    out_tokenizer = sp
    pad_id = out_tokenizer.pad_id()
    mask_id = out_tokenizer.piece_to_id(MASK)
    blank_id = out_tokenizer.piece_to_id(blank)

    logger.info(f"{blank_id=}, {mask_id=}, {pad_id=}")

    # Dataset
    test_input_files = config.data.test_input_files
    test_output_files = config.data.test_output_files
    dataset = DeBERTaAndDiTDataset(
        input_files=test_input_files,
        target_files=test_output_files,
        in_tokenizer=in_tokenizer,
        out_tokenizer=out_tokenizer,
    )
    logger.info(f"Building datasets: {test_input_files} / {test_output_files}")

    # DataLoader 생성 (collate_fn 지정)
    collator = DeBERTaAndDiTCollator(pad_id=dataset.pad_id)

    vocab_size = out_tokenizer.get_piece_size()
    batch_size = config.eval.batch
    pad_id = dataset.pad_id
    logger.info(f"vocab size: {vocab_size}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers
    )

    slu_model.eval()
    # evaluation
    for batch in loader:        
        input_ids = batch["input_ids"].to(device)
        input_masks = batch["input_masks"].to(device)
        target_ids = batch["target_ids"].to(device).long()
        target_masks = batch["target_masks"].to(device)

        ret = slu_model.greedy_search(
            x=input_ids,
            masks=input_masks,
        )

        # remove pad_id from output
        decoded = []
        for i in range(ret.shape[0]):
            tmp = ret[i, ret[i] != pad_id]
            decoded.append(sp.decode(tmp.tolist()))

        targets = []
        for i in range(target_ids.shape[0]):
            tmp = target_ids[i, target_ids[i] != pad_id]
            targets.append(sp.decode(tmp.tolist()))

        for i in range(len(decoded)):
            print(f"Target   : {targets[i]}")
            print(f"Decoded  : {decoded[i]}")
            print(f"Exact Matching: {targets[i]==decoded[i]}")
            print("-----")
        import sys
        sys.exit(0)


if __name__ == "__main__":
    main()