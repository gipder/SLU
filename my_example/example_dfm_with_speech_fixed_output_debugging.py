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
from tqdm import tqdm

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
from speech_featured_unet import DiscreteContextUnetConfig, DiscreteContextUnet
from model2 import DFMConfig, DFMModel
from my_length_predictor import ConvLengthPredictionModule
from length_predictor import LengthPredictor
from sampling import sampling, sampling_batch, sampling_debugging


class MyModelWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        emb_seq = extras["emb_seq"]
        emb_mask = extras["emb_mask"]
        logits = self.model(x, t, emb_seq, emb_mask) # B, T_out, K
        prob = torch.nn.functional.softmax(logits.float(), dim=-1)
        return prob

# -------------------------
# Train loop
# -------------------------
def train_dfm(
    dfm_model,
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
    from_scratch: bool = False,
    is_uniform: bool = False,
    mask_id: int = 1,
    debugging: bool = False,
):
    #os.makedirs(save_dir, exist_ok=True)
    from_scratch = True

    sp = train_loader.dataset.sp

    optim = torch.optim.AdamW(
        list(dfm_model.parameters()),
        lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(use_amp and device.startswith("cuda")))

    global_step = 0

    # a convex path path
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    audio_criterion = nn.CrossEntropyLoss(reduction="mean")
    text_criterion = nn.CrossEntropyLoss(reduction="mean")
    if from_scratch:
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = MixturePathGeneralizedKL(path)
    
    for ep in range(1, epochs+1):
        # train mode
        dfm_model.train()
        for batch in train_loader:

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

            assert T == x1_mask.sum(-1).max().item()

            # sample time t ~ Uniform(eps,1)
            t = torch.rand(B, device=device).clamp(1e-4, 1.0 - 1e-4)

            if is_uniform:
                # x0 ~ Uniform(0, K-1) , (B, T_out, K)
                x0 = torch.randint(0, K, (B, T), device=device)
            else:
                # x0 ~ Mask(0, K-1)
                x0 = torch.full_like(x1, mask_id, device=device)

            sample = path.sample(t=t, x_0=x0, x_1=x1)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=False):
                # logits B, T, K
                logits = dfm_model(x_t=sample.x_t, t=sample.t,
                                emb_feats=audio_feats,
                                emb_mask=audio_feat_mask
                                )

                if from_scratch:
                    corrupt_mask = (x1 != sample.x_t) # B, T_out
                    logits = logits.permute(0, -1, 1)
                    dit_loss = criterion(logits, x1)
                    mask = corrupt_mask.float()
                    denom = mask.sum().clamp_min(1.0)
                    dit_loss = (dit_loss * mask).sum() / denom
                    loss = dit_loss
                else:
                    dit_loss = criterion(logits=logits, x_1=sample.x_1, x_t=sample.x_t, t=sample.t)
                    loss = dit_loss

            #print(f"{target_lens=}")
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    list(dfm_model.parameters()),
                    grad_clip
                )

            scaler.step(optim)
            scaler.update()

            if global_step % log_every == 0:
                print(f"[ep {ep:02d} | step {global_step:06d}] "
                        f"loss={loss.item():.4f} "
                        f"unet_loss={dit_loss.item():.4f} ")
                #break # for debugging
            global_step += 1
        """
        # save each epoch
        ckpt_path = os.path.join(save_dir, f"model_step{ep:04d}.pt")
        torch.save(
            {
                "epoch": ep,
                "dfm_model": dfm_model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "K": K,
            },
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")
        """
        probability_denoiser = MyModelWrapper(dfm_model)

        if debugging is True:
            sampling_method = sampling_debugging
        else:
            sampling_method = sampling_batch

        hyps, targets = sampling_method(
            test_dl=test_loader,
            model=probability_denoiser,
            n_step=5,
            K=K,
            max_output_length=fixed_output_length,
            mask_id=mask_id,
            return_intermediates=True,
            is_uniform=False,
            device=device,
        )

        total = len(hyps)
        correct = 0
        for hyp, target in zip(hyps, targets):
            if hyp == target:
                correct += 1
        print(f"Epoch {ep} Exact Matching: {correct/total * 100:.4f} ({correct}/{total})")



    return

if __name__ == "__main__":
    # setting
    batch_size = 256
    test_batch_size = 1024
    K = 650
    D = 256
    D_f = 512
    lr = 0.0001
    n_head = 4
    max_out_len = 16
    n_layer = 2
    epochs = 3000
    bpe_model_path = "./bpe_model/bpe_650.model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HuBERTandDeBERTaDataset(
        task="test",
        bpe_file=bpe_model_path,
        debugging=False,
        debugging_num=128,
    )

    train_sampler = MyBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)

    train_dl = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        collate_fn=my_collate_fn,
    )

    total_train_data_count = len(train_dl.dataset)
    print(f"전체 train 데이터 개수: {total_train_data_count:,}")

    test_dataset = HuBERTandDeBERTaDataset(
        task="test",
        bpe_file=bpe_model_path,
        debugging=True,
        debugging_num=1024,
    )
    test_sampler = MyBatchSampler(test_dataset, batch_size=test_batch_size, shuffle=False)

    test_dl = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=4,
        collate_fn=my_collate_fn,
    )

    total_test_data_count = len(test_dl.dataset)
    print(f"전체 test 데이터 개수: {total_test_data_count:,}")

    unet_cfg = DiscreteContextUnetConfig(
        num_classes=K,
        n_feat=D,
        ctx_dim=D_f,
        n_heads=n_head
    )

    cfg = DFMConfig(unet_cfg)
    dfm_model = DFMModel(cfg, device=device)

    trainable_params = 0
    for model in [dfm_model.unet]:
        tmp_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params += tmp_trainable_params
        print(f"Trainable Parameters: {tmp_trainable_params:,}")
    print(f"Total Trainable Parameters: {trainable_params:,}")
    print(f"{dfm_model.device=}")

    # mask ID
    MASK = "[MASK]"
    sp = train_dataset.sp
    mask_id = sp.piece_to_id(MASK)
    train_dfm(
        dfm_model=dfm_model,
        K=K,
        train_loader=train_dl,
        test_loader=test_dl,
        epochs=epochs,
        lr=lr,
        save_dir=f"unet_dfm_speech_lr{lr}",
        fixed_output_length=max_out_len,
        mask_id=mask_id,
        debugging=True,
    )