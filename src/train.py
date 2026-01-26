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

# 1. LLM 모델과 토크나이저 로드 (microsoft/deberta-base 사용)
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
print(f"{pad_id=}, {mask_id=}")

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

# a convex path path
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)

# loss function
loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()
llambda = 0.1

# configure optimizer
lr = 1e-3
optim = torch.optim.AdamW(dit.parameters(), lr=lr)
epoch = 1000
batch_size = 256
save_step = 2000
num_remain_ckpt = 10
save_dir = "./exp/init"
task = "DeBERTa_DiT_SLU"
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
dit.train()

# print parameter count
dit_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
print(f"Total trainable parameters in DiT: {dit_params:,}")
length_predictor_params = sum(
    p.numel() for p in length_predictor.parameters() if p.requires_grad
    )
print(f"Total trainable parameters in Length Predictor: {length_predictor_params:,}")
print(f"Total parameters: {dit_params + length_predictor_params:,}")

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
        with torch.no_grad():
            text_embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["input_masks"]
            ).last_hidden_state.to(device)  # B x T x H

        # length prediction
        # lengths: B x T
        # length_logits: B x T
        lengths, length_logits = length_predictor(
            x=text_embeddings.transpose(0,1),  # T x B x H
            encoder_padding_mask=~batch["input_masks"].to(device).bool()  # B x T (True = pad)
        )

        # target length
        # predic lengths is argmax of predicted length logits
        target_lengths = batch["target_masks"].sum(dim=1).to(device)  # B,
        predic_lengths = lengths.argmax(dim=1)  # B,

        B = target_lengths.shape[0]
        T = target_lengths.max().item()
        C = vocab_size

        x_0 = torch.zeros((B, T)).to(device) # B x T
        x_0 = x_0 + mask_id  # set to <mask> token id
        x_1 = batch["target_ids"].to(device)  # B x T
        #x_0 = torch.zeros(B, 128, vocab_size)  # One-hot encoding
        t = torch.rand(B,).to(device)  # Dummy time embedd

        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t  # Noisy input at time t

        masks = batch["target_masks"].to(device)  # B x T mask

        # running decoder
        logits = dit(
            x_t=x_t,
            time=t,
            audio_embeddings=text_embeddings,
            preserve_mask=masks,
            audio_projected=None,
            audio_k_all=None,
            audio_v_all=None,
            cfg_strength=1.0,
            audio_drop_prob=0.1,
            use_gradient_checkpointing=False,
        )

        ## loss compute for MASK positions only
        x_1[x_t != mask_id] = -100
        x_1[x_1 == pad_id] = -100  # ignore pad positions in loss

        loss_dfm = loss_ce(
            input=logits.transpose(-1, -2),
            target=x_1,
        )

        loss_len = loss_ce(
            input=length_logits,
            target=target_lengths,
        )

        loss = loss_dfm + llambda * loss_len
        #print(f"{logits=}")
        #print(f"{loss_dfm=}")
        #print(f"{loss_len=}")
        #print(f"{loss=}")

        loss.backward()
        optim.step()
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
        "dit_state_dict": dit.state_dict(),
        "length_predictor_state_dict": length_predictor.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "step": step,
    }, save_path)
    print(f"Model saved at end of epoch {e+1} to {save_path}")

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
    
        