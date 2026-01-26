import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from dit import DiTSeq2SeqConfig, DiTSeq2Seq
from text_audio_fuse import TextAudioFuseConfig, TextAudioFusePool
from speech_featured_unet import DiscreteContextUnetConfig, DiscreteContextUnet

@dataclass
class DFMConfig:
    #dit: DiTSeq2SeqConfig
    #fusion: TextAudioFuseConfig
    unet: DiscreteContextUnetConfig

class DFMModel(nn.Module):
    def __init__(self, cfg: DFMConfig, device=None):        
        super().__init__()
        self.unet_cfg = cfg.unet        
        self.device = torch.device(device) if device else None

        self.unet = DiscreteContextUnet(self.unet_cfg)
        #self.fusion = TextAudioFusePool(self.fusion_cfg)        
        #self.dit = DiTSeq2Seq(self.dit_cfg)

        self.proj = nn.Linear(1024, 512)

        if self.device is not None:
            self.to(self.device)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        # condition
        emb_feats: torch.Tensor,
        emb_mask: torch.Tensor,
        #audio_feats: torch.Tensor,  # B, T_a, D
        #audio_mask: torch.Tensor,   # B, T_a
        #text_feats: torch.Tensor,   # B, T_t, D
        #text_mask: torch.Tensor,    # B, T_t    
        H: int = 4,
        W: int = 4,
    ) -> torch.Tensor:
        
        #B = x_t.size(0)
        #T_out = x_t.size(1)
        #dit_cfg = self.dit_cfg
        #fusion_cfg = self.fusion_cfg

        # fuse first
        # emb_pool can be used but sequential embeddings have more information
        # emb_seq: B, T, D
        # emb_mask: B, T
        """
        _, emb_seq, _ = self.fusion(
            text_feats, text_mask,
            audio_feats, audio_mask,
            need_attn_weights=True
        )
        """
        #emb_seq = audio_feats
        #emb_mask = audio_mask

        # x_t: B, T
        B = x_t.shape[0]
        T = x_t.shape[1]
        K = self.unet_cfg.num_classes
        if x_t.dim() == 2:
            assert T == H * W
            x_t = x_t.reshape(B, H, W)

        x_t = nn.functional.one_hot(x_t, K)
        x_t = x_t.permute(0, 3, 1, 2)

        emb_seq = self.proj(emb_feats) # B, T, D
        logits = self.unet(
            x_t, t, emb_seq, emb_mask
        )        
        logits = logits.permute(0, 2, 3, 1)
        logits = logits.reshape(B, T, K)

        return logits
    

if __name__ == "__main__":
    B = 2
    K = 650
    T_out = 16
    D = 1024    
    n_H = 8

    unet_cfg = DiscreteContextUnetConfig(
        num_classes=K,
        n_feat=D,
        ctx_dim=D//2,
        n_heads=n_H
    )    
    
    cfg = DFMConfig(unet_cfg)
    model = DFMModel(cfg)

    x_t = torch.randint(0, K, (B, T_out))
    t = torch.rand(B)

    audio_feats = torch.rand((B, T_out * 4, D))
    audio_mask = torch.ones(B, T_out * 4)

    text_feats = torch.rand((B, T_out * 2, D))
    text_mask = torch.ones(B, T_out * 2)
        
    logits = model(
        x_t, t,
        audio_feats, audio_mask,
        #text_feats, text_mask
    )

    print(f"input audio: {audio_feats.shape}")
    print(f"input text: {text_feats.shape}")
    print(f"logits: {logits.shape}")


