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
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from model_wrapper import WrappedModel

from sampling import sample

# 1. loading LLM for text embeddings
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# output tokenizer
sp = spm.SentencePieceProcessor()
sp.load("data/STOP_text/bpe_650.model")

in_tokenizer = tokenizer
out_tokenizer = sp
pad_id = out_tokenizer.pad_id() if out_tokenizer.pad_id() != -1 else 0
mask_id = out_tokenizer.piece_to_id("[MASK]")

# Dataset
dataset = DeBERTaAndDiTDataset(
    input_files="data/STOP_text/low.eval.asr",
    target_files="data/STOP_text/low.eval.slu",
    in_tokenizer=in_tokenizer,
    out_tokenizer=out_tokenizer,
)

# DataLoader 생성 (collate_fn 지정)
collator = DeBERTaAndDiTCollator(pad_id=dataset.pad_id)

# DFM Decoder DiT
# configuration for DiT
vocab_size = out_tokenizer.get_piece_size()
print(f"{vocab_size=}")
config = {
    "hidden_size": 512,
    "n_heads": 8,
    "cond_dim": vocab_size,
    "n_blocks": 6,
    "dropout": 0.1,
    "whisper_dim": 768, # DeBERTa embedding dim
    "pad_id": dataset.pad_id,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dit = DitTransformer(
    vocab_size=vocab_size,
    masked=False, # already handled in dataset
    config=config)
dit = dit.to(device)

# Length predictor
# Conv first
max_length = 128
lp_config = {
    "embed_dim": config["whisper_dim"],
    "conv_dim": config["hidden_size"],
    "max_target_positions": max_length,
}
length_predictor = ConvLengthPredictionModule(
    embed_dim=lp_config["embed_dim"],
    conv_dim=lp_config["conv_dim"],
    max_target_positions=lp_config["max_target_positions"],
)
length_predictor = length_predictor.to(device)

# load trained model
batch_size = 1
epoch=100
prefix="DeBERTa_DiT_SLU"
ckpt = torch.load(f"exp/init/{prefix}_epoch{epoch}.pt", map_location=device)
dit.load_state_dict(ckpt["dit_state_dict"])
length_predictor.load_state_dict(ckpt["length_predictor_state_dict"])

dit.eval()
length_predictor.eval()

# Eval loop
dit_wrapper = WrappedModel(dit)
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)
mask_id = sp.piece_to_id("[MASK]")
pad_id = sp.pad_id() if sp.pad_id() != -1 else 0

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator,
    num_workers=4
)

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
        print(f"{batch['target_ids']=}")
        print(f"{batch['target_ids'].shape=}")        
        # masks from predicted length
        predicted_lengths = lengths.argmax(dim=1) + 1 # B,
        #oracle_lengths = (batch['target_ids'] != pad_id).sum(dim=1).to(device)
        #predicted_lengths = oracle_lengths
        #print(f"{oracle_lengths=}")
        #print(f"{predicted_lengths=}")
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
        # retrieval from input_ids
        input_text = in_tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        target_text = out_tokenizer.decode(batch['target_ids'][0].tolist())
        output_text = sp.decode(samples[0].tolist())
        print(f"{input_text=}")        
        print(f"{target_text=}")
        print(f"{output_text=}")
        break
