import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from dit import DiTSeq2SeqConfig, DiTSeq2Seq
from text_audio_fuse import TextAudioFuseConfig, TextAudioFusePool




@dataclass
class DFMConfig:
    dit: DiTSeq2SeqConfig
    fusion: TextAudioFuseConfig

class DFMModel(nn.Module):
    def __init__(self, cfg: DFMConfig, device=None):        
        super().__init__()
        self.dit_cfg = cfg.dit
        self.fusion_cfg = cfg.fusion

        self.device = torch.device(device) if device else None

        self.fusion = TextAudioFusePool(self.fusion_cfg)        
        self.dit = DiTSeq2Seq(self.dit_cfg)

        self.proj = nn.Linear(1024, 512)

        if self.device is not None:
            self.to(self.device)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        # condition
        audio_feats: torch.Tensor,  # B, T_a, D
        audio_mask: torch.Tensor,   # B, T_a
        text_feats: torch.Tensor,   # B, T_t, D
        text_mask: torch.Tensor,    # B, T_t    
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
        emb_seq = audio_feats
        emb_mask = audio_mask

        emb_seq = self.proj(emb_seq)

        logits = self.dit(
            x_t, t, emb_seq, emb_mask
        )

        return logits
    

if __name__ == "__main__":
    B = 2
    K = 650
    T_out = 8
    D = 256    
    dit_cfg = DiTSeq2SeqConfig(K=650, max_T_out=T_out,
                               d_model=D, cond_in_dim=D)
    fusion_cfg = TextAudioFuseConfig(d_model=D, d_out=D)

    cfg = DFMConfig(dit_cfg, fusion_cfg)
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
        text_feats, text_mask
    )

    print(f"input audio: {audio_feats.shape}")
    print(f"input text: {text_feats.shape}")
    print(f"logits: {logits.shape}")


