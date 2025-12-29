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

class ConvUpsample(nn.Module):
    def __init__(self, d_model, upsample_factor):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=upsample_factor,
            stride=1,
            padding=upsample_factor // 2,
            groups=d_model,   # depthwise
        )

    def forward(self, x):
        # x: B x T x D
        x = x.transpose(1, 2)        # B x D x T
        x = self.conv(x)             # B x D x T
        x = x.repeat_interleave(upsample_factor, dim=-1)
        return x.transpose(1, 2)     # B x (T*upsample) x D

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

d_model = 768
upsample_factor = 8
proj = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.LayerNorm(d_model),    
    nn.Linear(d_model, vocab_size)
).to(device)

upsample = ConvUpsample(d_model, upsample_factor).to(device)

# loss function
loss_ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

# configure optimizer
lr = 1e-3
optim = torch.optim.AdamW(list(proj.parameters()) + list(upsample.parameters()),
                          lr=lr,
                          weight_decay=1e-4)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

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
proj_params = sum(p.numel() for p in proj.parameters() if p.requires_grad)
print(f"Total trainable parameters in the projection: {proj_params:,}")

# encoder is freezed
model.eval()

# training loop
step = 0
for e in range(epoch):
    print(f"Epoch {e+1}/{epoch} started.")
    for batch in loader:
        optim.zero_grad()
        #print("Input Shape:", batch["input_ids"].shape)
        #print("Input Ids:", batch["input_ids"])
        #print("Input Mask:", batch["input_mask"][0]) # 첫 번째 샘플 마스크 확인
        #print("Target Ids:", batch["target_ids"]) # 첫 번째 샘플 마스크 확인

        # running encoder
        with torch.inference_mode():
            text_embeddings = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["input_masks"].to(device)
            ).last_hidden_state.to(device)  # B x T x H

        B, T, D = text_embeddings.shape
        
        text_embeddings = text_embeddings.unsqueeze(2).repeat(1, 1, upsample_factor, 1)
        text_embeddings = text_embeddings.reshape(B, T * upsample_factor, D)
        text_embeddings = upsample(text_embeddings)
        
        targets = batch["target_ids"].to(device).long()  # B x T
        masks = batch["target_masks"].to(device)  # B x T mask
        target_lengths = masks.sum(dim=-1).long() # B,
        
        input_lengths = batch["input_masks"].sum(dim=-1).long() * upsample_factor
        input_lengths = input_lengths.to(device)

        bad = (target_lengths > input_lengths)
        if bad.any():
            print("CTC impossible samples:", bad.nonzero().squeeze(-1)[:20])
            print("input_lengths:", input_lengths[bad][:20])
            print("target_lengths:", target_lengths[bad][:20])

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = proj(text_embeddings)                
            log_probs = logits.log_softmax(dim=-1)
            log_probs = log_probs.transpose(0, 1)

            loss = loss_ctc(log_probs, targets, input_lengths, target_lengths)

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        nn.utils.clip_grad_norm_(upsample.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()        
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        """
        if step % save_step == 0 and step > 0:
            save_path = os.path.join(save_dir, f"{task}_step{step}.pt")
            torch.save({
                "dit_state_dict": dit.state_dict(),
                "length_predictor_state_dict": length_predictor.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "step": step,
            }, save_path)
            print(f"Model saved at step {step} to {save_path}")
        """
        step += 1

    save_path = os.path.join(save_dir, f"{task}_epoch{e+1}.pt")
    torch.save({
        "proj_state_dict": proj.state_dict(),        
        "optimizer_state_dict": optim.state_dict(),
        "step": step,
    }, save_path)
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
    
        