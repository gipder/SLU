import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from length_predictor import MaskedLengthPredictionModule

@dataclass
class LengthModelConfig:
    # Length Predictor 설정
    embed_dim: int = 1024
    length_hidden_dim: int = 512
    max_target_positions: int = 256
    length_dropout: float = 0.1
    length_condition: str = "text"  # "audio" or "text"
   

class LengthModel(nn.Module):
    def __init__(self, cfg: LengthModelConfig):
        super().__init__()
        self.cfg = cfg        
        self.length_predictor = MaskedLengthPredictionModule(
            embed_dim=cfg.embed_dim,
            length_hidden_dim=cfg.length_hidden_dim,
            max_target_positions=cfg.max_target_positions,
            length_dropout=cfg.length_dropout,            
        )

    def forward(
        self,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        text_feats: torch.Tensor = None,
        text_mask: torch.Tensor = None,        
    ) -> torch.Tensor:
        if self.cfg.length_condition == "text":
            logits = self.length_predictor(
                text_feats, ~(text_mask.bool())
            )
        elif self.cfg.length_condition == "audio":
            logits = self.length_predictor(
                audio_feats, ~(audio_mask.bool())
            )
        else:
            raise ValueError(f"Unknown length_condition: {self.cfg.length_condition}")

        return logits


if __name__ == "__main__":
    B = 2
    K = 650
    T_out = 16
    D = 1024
    n_H = 8

    cfg = LengthModelConfig(
        embed_dim=D,
        length_hidden_dim=512,
        max_target_positions=256,
        length_dropout=0.1,
        length_condition="text"
    )    

    print(f"{cfg=}")
    model = LengthModel(cfg)

    audio_feats = torch.rand((B, T_out * 4, D))
    audio_mask = torch.ones(B, T_out * 4).bool()

    text_feats = torch.rand((B, T_out * 2, D))
    text_mask = torch.ones(B, T_out * 2).bool()

    logits = model(
        audio_feats, audio_mask,
        text_feats, text_mask
    )

    print(f"input audio: {audio_feats.shape}")
    print(f"input text: {text_feats.shape}")
    print(f"logits: {logits.shape}")
    print(f"length_logits: {logits.shape}")



