# DFM example with DeBERTa model + DiT
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from my_dataset import Seq2SeqCollator, Seq2SeqDataset
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from deberta_and_dit_dataset import DeBERTaAndDiTDataset, DeBERTaAndDiTCollator
from drax.transformer import DitTransformer
from my_length_predictor import ConvLengthPredictionModule, MaskedLengthPredictionModule
import os

# For DFM
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
#from flow_matching.solver import MixtureDiscreteEulerSolver
#from flow_matching.utils import ModelWrapper
#from flow_matching.loss import MixturePathGeneralizedKL
from model_wrapper import WrappedModel

# For DFM sampmling
from sampling import sample

# Config
from slu_model import SLUConfig, SLUModel
from upsampler import LearnedUpsample
from projector import CTCProjector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LLM 모델과 토크나이저 로드 (microsoft/deberta-base 사용)
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# output tokenizer
sp = spm.SentencePieceProcessor()
sp.load("data/STOP_text/bpe_650.model")

blank = "<blk>"
in_tokenizer = tokenizer
out_tokenizer = sp
pad_id = out_tokenizer.pad_id()
mask_id = out_tokenizer.piece_to_id("[MASK]")
blank_id = out_tokenizer.piece_to_id("<blk>")

print(f"{blank_id=}, {mask_id=}, {pad_id=}")

# Dataset
dataset = DeBERTaAndDiTDataset(
    input_files="data/STOP_text/low.eval.asr",
    target_files="data/STOP_text/low.eval.slu",
    in_tokenizer=in_tokenizer,
    out_tokenizer=out_tokenizer,
)

# DataLoader 생성 (collate_fn 지정)
collator = DeBERTaAndDiTCollator(pad_id=dataset.pad_id)

vocab_size = out_tokenizer.get_piece_size()
print(f"{vocab_size=}")
print(f"{out_tokenizer.piece_to_id('<blk>')=}")

config = SLUConfig(
    llm_name=model_name,
    upsample_factor=8,
    d_model=768,
    blank_token_id=blank_id,
    vocab_size=vocab_size,
)

proj = CTCProjector(
    input_dim=config.d_model,
    d_model=config.d_model,
    vocab_size=config.vocab_size,
).to(device)

upsample = LearnedUpsample(
    d_model=config.d_model,
    upsample_factor=config.upsample_factor,
).to(device)

slu_model = SLUModel(
    config=config,
    upsampler=upsample,
    proj=proj,
).to(device)

# configure optimizer
lr = 1e-3
weight_decay = 1e-4
optim = torch.optim.AdamW(slu_model.parameters(),
                          lr=lr,
                          weight_decay=weight_decay)

scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

epoch = 100
batch_size = 256
save_step = 2000
num_remain_ckpt = 10
save_dir = "./exp/ctc_loss"
task = "DeBERTa_CTC_SLU"

# if save dir does not exist, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator,
    num_workers=4
)

# print parameter count
params = sum(p.numel() for p in slu_model.parameters() if p.requires_grad)
print(f"Total trainable parameters in the projection: {params:,}")

# training loop
step = 0
for e in range(epoch):
    print(f"Epoch {e+1}/{epoch} started.")
    for batch in loader:
        optim.zero_grad()

        input_ids = batch["input_ids"].to(device)
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
            print(f"Step {step}, Loss: {loss.item()}")
        step += 1

    save_path = os.path.join(save_dir, f"{task}_epoch{e+1}.pt")
    slu_model.save_checkpoint(
        path=save_path,
        optimizer=optim,
        step=step,
    )    
    print(f"Model saved at end of epoch {e+1} to {save_path}")

import sys
sys.exit(0)

# sampling example
dit.eval()

dit_wrapper = WrappedModel(dit)
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)
mask_id = sp.piece_to_id("[MASK]")
pad_id = sp.pad_id() if sp.pad_id() != -1 else 0

with torch.no_grad():
    for batch in loader:
        text_embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["input_masks"]
            ).last_hidden_state.to(device)  # B x T x H
        lengths, length_logits = length_predictor(
            x=text_embeddings.transpose(0,1),  # T x B x H
            encoder_padding_mask=~batch["input_masks"].to(device).bool()  # B x T (True = pad)
        )
        # masks from predicted length
        predicted_lengths = lengths.argmax(dim=1) + 1 # B,
        B = predicted_lengths.shape[0]
        T = predicted_lengths.max().item()
        masks = torch.arange(T).unsqueeze(0).repeat(B,1).to(device) < predicted_lengths.unsqueeze(-1) # B x T
        
        samples = sample(
            model_wrapper=dit_wrapper,
            text_embeddings=text_embeddings,
            embedding_masks=masks,
            predited_lengths=predicted_lengths,
            vocab_size=vocab_size,
            mask_id=mask_id,
            pad_id=pad_id,
            path=path,
            steps=10,
        )
        print("samples:", samples)
        break
    
        