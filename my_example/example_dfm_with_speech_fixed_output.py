import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import librosa
import glob
from transformers import AutoProcessor, HubertForCTC
from torchvision import models, transforms
from torchvision.datasets import MNIST
from transformers import HubertModel
from torch.nn.utils.rnn import pad_sequence
#from speech_featured_unet import DiscreteContextUnet
from torch.amp import autocast, GradScaler

# For DFML
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver

# my implementation
from utils import make_gif_from_xts
from my_dataset import HuBERTandDeBERTaDataset, MyBatchSampler
from my_dataset import my_collate_fn
from text_audio_fuse import TextAudioFusePool
from dit import DiTSeq2SeqConfig, DiTSeq2Seq
from my_length_predictor import ConvLengthPredictionModule
from length_predictor import LengthPredictor
# -------------------------
# Train loop
# -------------------------
def train_dfm(
    fusion_model,
    dit_model,
    len_pred_model,
    K: int,
    train_loader,
    test_loader: Optional[DataLoader],
    device="cuda",
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    log_every: int = 100,
    save_dir: str = "./ckpt_dfm",
    fixed_output_length: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)
    fusion_model.to(device)
    dit_model.to(device)
    len_pred_model.to(device)

    sp = train_loader.dataset.sp

    optim = torch.optim.AdamW(
        list(fusion_model.parameters())
        +list(dit_model.parameters())
        +list(len_pred_model.parameters()), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(use_amp and device.startswith("cuda")))

    global_step = 0

    # a convex path path
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    from_scratch = False
    if from_scratch:
        criterion = nn.CrossEntropyLoss(reduction="none")
        length_criterion = nn.CrossEntropyLoss(reduction="mean")
    else:
        criterion = MixturePathGeneralizedKL(path)
        length_criterion = nn.CrossEntropyLoss(reduction="mean")

    # train mode
    fusion_model = fusion_model.to(device)
    fusion_model.train()
    dit_model = dit_model.to(device)
    dit_model.train()
    len_pred_model = len_pred_model.to(device)
    len_pred_model.train()

    for ep in range(1, epochs + 1):                
        for it, batch in enumerate(train_loader):
            # batch: (audio_feat, audio_feat_mask, text_feat, text_feat_mask,)
            (
                audio_feats, audio_feat_mask,
                text_feats, text_feat_mask,
                slus, slu_mask
             ) = batch

            # x1: B, T_o
            # dtype/shape 정리
            audio_feats = audio_feats.to(device) # B, T, D
            audio_feat_mask = audio_feat_mask.to(device)
            text_feats = text_feats.to(device)
            text_feat_mask = text_feat_mask.to(device)
            x1 = slus.to(device)
            x1_mask = slu_mask.to(device)

            B = x1.size(0)
            T = x1.size(1)
            D = x1.size(-1)

            assert T == x1_mask.sum(-1).max().item()
            if T < fixed_output_length:
                new_T = fixed_output_length
                new_x1 = torch.zeros((B, new_T, D), device=device).long()
                new_x1[B, :T, :] = x1

                new_x1_mask = torch.zeros((B, new_T), device=device).bool()
                new_x1_mask[B, :T] = x1_mask

                x1 = new_x1
                x1_mask = new_x1_mask
            #print(f"{x1=}")
            #print(f"{x1_mask=}")
            #import sys
            #sys.exit(0)

            #fusion first
            _, emb_seq, _ = fusion_model(
                text_feats, text_feat_mask,
                audio_feats, audio_feat_mask,
                need_attn_weights=True
            )
            emb_mask = text_feat_mask

            """
            # len_pred: B, MAX_T + 1 (becasue of 0)
            len_pred_logits = len_pred_model(
                emb_seq, emb_mask
            )
            target_lens = x1_mask.sum(-1).long()
            """
            # sample time t ~ Uniform(eps,1)
            t = torch.rand(B, device=device).clamp(1e-4, 1.0 - 1e-4)

            # masking or uniform
            is_uniform = False
            # x0 ~ Uniform(0, K-1) , (B, T_out, K)
            mask_symbol = "[MASK]"
            mask_id = sp.piece_to_id(mask_symbol)            
            if is_uniform:
                x0 = torch.randint(0, K, (B, T), device=device)
            else:
                # x0 ~ Mask(0, K-1)
                x0 = torch.full_like(x1, mask_id, device=device)

            sample = path.sample(t=t, x_0=x0, x_1=x1)

            corrupt_mask = (x1 != sample.x_t) # B, T_out

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=False):
                # logits B, T, K
                logits = dit_model(sample.x_t, sample.t,
                                   emb_seq, emb_mask)
                alpha = 1.0
                if from_scratch:
                    logits = logits.permute(0, -1, 1)                    
                    dit_loss = criterion(logits, x1)                    
                    mask = corrupt_mask.float()                    
                    denom = mask.sum().clamp_min(1.0)
                    dit_loss = (dit_loss * mask).sum() / denom
     
                    len_loss = torch.tensor(0.0) # length_criterion(len_pred_logits, target_lens)                    
                    loss = dit_loss + alpha * len_loss
                else:
                    len_loss = torch.tensor(0.0) # length_criterion(len_pred_logits, target_lens)                    
                    dit_loss = criterion(logits=logits, x_1=sample.x_1, x_t=sample.x_t, t=sample.t)                    
                    loss = dit_loss + alpha * len_loss                
                
            #print(f"{target_lens=}")
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    list(fusion_model.parameters())
                    +list(dit_model.parameters())
                    +list(len_pred_model.parameters()),
                    grad_clip
                )

            scaler.step(optim)
            scaler.update()            

            if global_step % log_every == 0:
                print(f"[ep {ep:02d} | step {global_step:06d}] "
                      f"loss={loss.item():.4f} "
                      f"dit_loss={dit_loss.item():.4f} "
                      f"len_loss={len_loss.item():.4f}")
                #break # for debugging
            global_step += 1

        # save each epoch
        ckpt_path = os.path.join(save_dir, f"model_ep{ep:03d}.pt")
        torch.save(
            {
                "epoch": ep,
                "fusion_model": fusion_model.state_dict(),
                "dit_model": dit_model.state_dict(),
                "len_pred_model": len_pred_model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "K": K,
            },
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")        

        # sampling
        eval_method = "closed"
        NUM = 1
        FIX_SAMPLE=200
        if eval_method == "closed":
            # 1개일 때만 테스트 NUM == 1
            test_batch = train_dl.dataset[FIX_SAMPLE]
            sample_audio_feats = test_batch[0].unsqueeze(0).to(device)
            sample_audio_feat_mask = test_batch[1].unsqueeze(0).to(device)
            sample_text_feats = test_batch[2].unsqueeze(0).to(device)
            sample_text_feat_mask = test_batch[3].unsqueeze(0).to(device)
            sample_slus = test_batch[-2].unsqueeze(0).to(device)
            sample_slu_mask = test_batch[-1].unsqueeze(0).to(device)

            # length
            #pred_lens = sample_slu_mask.sum(-1)
            T = fixed_output_length #pred_lens.max().item()

            # x0: B, T_out
            if is_uniform:
                x_0 = torch.randint(0, K, (NUM, T), device=device)
            else:
                x_0 = torch.full((NUM, T), mask_id, device=device)
            # emb_seq, emb_mask
            _, sample_emb_seq, _ = fusion_model(
                sample_text_feats, sample_text_feat_mask,
                sample_audio_feats, sample_audio_feat_mask,
                need_attn_weights=True
            )
            sample_emb_mask = sample_text_feat_mask
            pred_length_logits = len_pred_model(sample_emb_seq, sample_emb_mask)
            pred_lengths = pred_length_logits.topk(4, dim=-1)
            print(f"Target length: {T}")
            print(f"Predicted length: {pred_lengths}")

        t = 0.0

        class MyModelWrapper(ModelWrapper):
            def __init__(self, model):
                super().__init__(model)

            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                emb_seq = extras["emb_seq"]
                emb_mask = extras["emb_mask"]
                logits = self.model(x, t, emb_seq, emb_mask) # B, T_out, K
                prob = torch.nn.functional.softmax(logits.float(), dim=-1)
                return prob

        probability_denoiser =MyModelWrapper(dit_model)

        solver = MixtureDiscreteEulerSolver(
            model=probability_denoiser,
            path=path,
            vocabulary_size=K,
        )

        n_steps = 10
        time_grid = torch.linspace(0.0, 1.0 - 1e-3, n_steps + 1, device=device)
        step_size = 1 / 100
        x_1_hat = solver.sample(
            x_init=x_0,
            step_size=step_size,
            time_grid=time_grid,
            return_intermediates=True,
            emb_seq=sample_emb_seq,
            emb_mask=sample_emb_mask,
        )
        x_1 = sample_slus

        sp = train_dl.dataset.sp
        results = []
        pieces = []
        for i in range(x_1_hat.shape[0]):
            #print(f"{x_1_hat[i, :NUM].tolist()=}")
            ids = x_1_hat[i, :NUM].tolist()[0]
            sentence = sp.decode(ids)
            sentence_ids = sp.id_to_piece(ids)
            #print(f"{sentence[:NUM]=}")
            results.append(sentence)
            pieces.append(", ".join(sentence_ids))

        target_ids =x_1[0].tolist()
        target_sentence = sp.decode(target_ids)
        target_sentence_ids = sp.id_to_piece(target_ids)
        for i in range(x_1_hat.shape[0]):
            print(f"* Step {i}")
            print(f"  Sentence: {results[i]}")
            print(f"  Tokens: {pieces[i]}")
        print(f"TARGET: {target_sentence}")
        print(f"TARGET ID: {', '.join(target_sentence_ids)}")
        print(f"Predicted T: {T}")
        print(f"Target T: {len(x_1[0].tolist())}")
        #list_xt = []
        #for i in range(x_1.shape[0]):
        #    list_xt.append(x_1[i])
        #make_gif_from_xts(list_xt, NUM, out_path=f"{save_dir}/stacked_ep{ep}.gif", mode="stack")

    return

if __name__ == "__main__":
    # setting
    batch_size = 64
    K = 650
    D = 1024
    lr = 0.0001
    n_head = 4
    max_out_len = 10
    n_layer = 2
    epochs = 300

    train_dataset = HuBERTandDeBERTaDataset(
        task="eval_0",
        bpe_file="./bpe_model/bpe_650.model"
    )
    train_sampler = MyBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)

    train_dl = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=my_collate_fn,
    )

    total_train_data_count = len(train_dl.dataset)
    print(f"전체 train 데이터 개수: {total_train_data_count:,}")

    """
    test_dataset = HuBERTandDeBERTaDataset(task="eval")
    test_sampler = MyBatchSampler(train_dataset, batch_size=batch_size, shuffle=False)

    test_dl = DataLoader(
        train_dataset,
        batch_sampler=test_sampler,
        num_workers=0,
        collate_fn=my_collate_fn,
    )

    total_test_data_count = len(test_dl.dataset)
    print(f"전체 test 데이터 개수: {total_test_data_count:,}")
    """
    fusion_model = TextAudioFusePool(
        d_model=D,
        d_ff=D,
        d_out=D//2,
        n_heads=n_head//2,
        n_layers=n_layer//2,
        pool="mean",
        is_proj=True
    )

    cfg = DiTSeq2SeqConfig(
        K=K,
        d_model=D//2,
        n_layers=n_layer,
        n_heads=n_head,
        max_T_out=max_out_len,
        cond_in_dim=D//2
    )
    dit_model = DiTSeq2Seq(cfg)
    """
    len_pred_model = ConvLengthPredictionModule(
        embed_dim=D//2,
        conv_dim=D//2,
        max_target_positions=max_out_len,
        length_dropout=0.2,
        glu=True,
        activation="glu",
        pooling_type="mean",
        kernel_sizes=[3, 5],
    )
    """
    len_pred_model = LengthPredictor(d_model=D//2, Lmax=max_out_len)

    trainable_params = 0
    for model in [fusion_model, dit_model, len_pred_model]:
        tmp_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params += tmp_trainable_params
        print(f"Trainable Parameters: {tmp_trainable_params:,}")
    print(f"Total Trainable Parameters: {trainable_params:,}")

    train_dfm(
        fusion_model=fusion_model,
        dit_model=dit_model,
        len_pred_model=len_pred_model,
        K=K,
        train_loader=train_dl,
        test_loader=train_dl,
        epochs=epochs,
        lr=lr,
        save_dir=f"unet_dfm_speech_lr{lr}"
    )