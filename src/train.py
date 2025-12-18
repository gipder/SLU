# DFM example with DeBERTa model + DiT
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from my_dataset import Seq2SeqCollator, Seq2SeqDataset
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from deberta_and_dit_dataset import DeBERTaAndDiTDataset, DeBERTaAndDiTCollator
from drax.transformer import DitTransformer

# 1. 모델과 토크나이저 로드 (microsoft/deberta-base 사용)
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

batch_size = 256
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collator,
    num_workers=4
)

# Decoder DiT
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

for batch in loader:
    #print("Input Shape:", batch["input_ids"].shape)    
    print("Input Ids:", batch["input_ids"])    
    #print("Input Mask:", batch["input_mask"][0]) # 첫 번째 샘플 마스크 확인
    print("Target Ids:", batch["target_ids"]) # 첫 번째 샘플 마스크 확인
    
    # running encoder
    text_embeddings = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["input_masks"]
    ).last_hidden_state.to(device)  # B x T x C

    B = text_embeddings.shape[0]
    T = text_embeddings.shape[1]
    C = text_embeddings.shape[2]
    
    # length predictor isn't used here
    x_0 = torch.zeros_like(batch["target_ids"]).to(device)  # B x T
    x_0 = x_0 + pad_id  # set to <mask> token id
    #x_0 = torch.zeros(B, 128, vocab_size)  # One-hot encoding
    t = torch.rand(B,).to(device)  # Dummy time embedd
    masks = batch["target_masks"].to(device)  # B x T mask    
    print(f"{masks=}")

    # running decoder
    logits = dit(
        x_t=x_0,
        time=t,
        audio_embeddings=text_embeddings,
        preserve_mask = masks,
        audio_projected=None,
        audio_k_all=None,
        audio_v_all=None,
        cfg_strength=1.0,
        audio_drop_prob=0.1,
        use_gradient_checkpointing=False,
    )

    print("Logits Shape:", logits.shape)
    print(f"{x_0=}")
    print(f"{logits=}")
    
    # 모델 입력 예시
    # logits = model(
    #     x_t=batch["target_ids"],    # Noisy Target (Train 시에는 noise 추가 로직 필요)
    #     cond=batch["input_ids"],    # Condition Text
    #     padding_mask=batch["target_mask"],
    #     cond_mask=batch["input_mask"]
    # )
    break
