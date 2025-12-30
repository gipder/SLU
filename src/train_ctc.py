import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging
import argparse
from typing import Optional

from transformers import AutoTokenizer, AutoModel
import sentencepiece as spm

# my implementation
from deberta_and_dit_dataset import DeBERTaAndDiTDataset, DeBERTaAndDiTCollator
from utils import setup_logger, get_config

# Config
from slu_model import SLUConfig, SLUModel
from upsampler import LearnedUpsample
from projector import CTCProjector
from config import ExperimentConfig, SLUConfig, MODEL_CONFIG_REGISTRY

#def main(config: OmegaConf, logger: Optional[logging.Logger]=None):    
def main():
    config = get_config()
    save_dir = config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = setup_logger(save_dir)
    logger.info(f"configuration information: {config}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LLM 모델과 토크나이저 로드 (microsoft/deberta-base 사용)
    model_name = config.model.llm_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModel.from_pretrained(model_name).to(device)
    logger.info(f"Loading {model_name} completed")

    # output tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(config.data.bpe_model_path)
    logger.info(f"Loading {config.data.bpe_model_path} completed")

    blank = config.data.blank
    MASK = config.data.MASK
    in_tokenizer = tokenizer
    out_tokenizer = sp
    pad_id = out_tokenizer.pad_id()
    mask_id = out_tokenizer.piece_to_id(MASK)
    blank_id = out_tokenizer.piece_to_id(blank)

    logger.info(f"{blank_id=}, {mask_id=}, {pad_id=}")

    # Dataset
    train_input_files = config.data.train_input_files
    train_output_files = config.data.train_output_files
    dataset = DeBERTaAndDiTDataset(
        input_files=train_input_files,
        target_files=train_output_files,
        in_tokenizer=in_tokenizer,
        out_tokenizer=out_tokenizer,
    )
    logger.info(f"Building datasets: {train_input_files} / {train_output_files}")

    # DataLoader 생성 (collate_fn 지정)
    collator = DeBERTaAndDiTCollator(pad_id=dataset.pad_id)

    vocab_size = out_tokenizer.get_piece_size()
    logger.info(f"vocab size: {vocab_size}")        

    MODEL_CONFIG = MODEL_CONFIG_REGISTRY[config.model.type]
    logger.info(f"ModelConfig: {MODEL_CONFIG}")
    
    # make configuration
    upsample_factor = config.model.upsample_factor
    d_model = config.model.d_model
    model_config = MODEL_CONFIG(
        llm_name=model_name,
        upsample_factor=upsample_factor,
        d_model=d_model,
        blank_token_id=blank_id,
        vocab_size=vocab_size,
    )
    logger.info(f"Model Config: {config}")

    proj = CTCProjector(
        input_dim=d_model,
        d_model=d_model,
        vocab_size=vocab_size,
    ).to(device)
    logger.info(f"Loading {proj} completed")

    upsample = LearnedUpsample(
        d_model=d_model,
        upsample_factor=upsample_factor,
    ).to(device)
    logger.info(f"Loading {upsample} completed")

    slu_model = SLUModel(
        config=config,
        upsampler=upsample,
        proj=proj,
    ).to(device)
    logger.info(f"Loading SLU model - {slu_model} completed")

    # configure optimizer
    lr = config.lr
    weight_decay = config.weight_decay
    optim = torch.optim.AdamW(slu_model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    end_epoch = config.end_epoch
    start_epoch = config.start_epoch
    assert start_epoch > 0

    batch_size = config.batch_size
    #save_step = 2000
    #num_remain_ckpt = 10
    save_dir = config.save_dir
    task =config.exp_name

    # if save dir does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers
    )

    # print parameter count
    params = sum(p.numel() for p in slu_model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters in the SLU model: {params:,}")

    # training loop
    step = 0
    for e in range(start_epoch, end_epoch+1):
        logger.info(f"Epoch {e}/{end_epoch} started.")
        for batch in loader:
            optim.zero_grad()

            input_ids = batch["input_ids"].to(device).long()
            input_masks = batch["input_masks"].to(device)
            target_ids = batch["target_ids"].to(device).long()
            target_masks = batch["target_masks"].to(device)

            ret = slu_model(
                input_ids=input_ids,
                input_masks=input_masks,
                target_ids=target_ids,
                target_masks=target_masks,
            )

            loss = ret['loss']

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            nn.utils.clip_grad_norm_(upsample.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            if step % 10 == 0:
                logger.info(f"Step {step}, Loss: {loss.item()}")
            step += 1

        save_path = os.path.join(save_dir, f"{task}_epoch{e}.pt")
        slu_model.save_checkpoint(
            path=save_path,
            optimizer=optim,
            step=step,
        )
        logger.info(f"Model saved at end of epoch {e} to {save_path}")

if __name__ == "__main__":
    main()