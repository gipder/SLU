import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from dit import DiscreteDualDiT
from length_predictor import MaskedLengthPredictionModule
from flow_matching.utils import ModelWrapper

@dataclass
class DFMModelConfig:
    # DIT 설정
    vocab_size: int = 42
    hidden_size: int = 512
    depth: int = 6
    num_heads: int = 8
    audio_dim: int = 1024
    text_dim: int = 1024
    # Length Predictor 설정
    embed_dim: int = audio_dim
    length_hidden_dim: int = 512
    max_target_positions: int = 128
    length_dropout: float = 0.1
    length_condition: str = "text"  # "audio" or "text"
    

class DFMModelWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        audio_feats = extras["audio_feats"]
        audio_mask = extras["audio_mask"]
        text_feats = extras["text_feats"]
        text_mask = extras["text_mask"]        
        logits, _ = self.model(x, t, audio_feats, audio_mask, text_feats, text_mask) # B, T_out, K
        prob = torch.nn.functional.softmax(logits.float(), dim=-1)
        return prob
    
    def get_length_logits(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        length_logits = self.model.length_predictor(x, ~(x_mask.bool()))
        #print(f"{length_logits.argmax(dim=-1)=}")
        return length_logits        


class DFMModel(nn.Module):
    def __init__(self, cfg: DFMModelConfig, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device) if device else None
        self.dit = DiscreteDualDiT(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            audio_dim=cfg.audio_dim,
            text_dim=cfg.text_dim
        )
        self.length_predictor = MaskedLengthPredictionModule(
            embed_dim=cfg.embed_dim,
            length_hidden_dim=cfg.length_hidden_dim,
            max_target_positions=cfg.max_target_positions,
            length_dropout=cfg.length_dropout
        )

        if self.device is not None:
            self.to(self.device)        

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,        # condition
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        text_feats: torch.Tensor = None,
        text_mask: torch.Tensor = None,        
    ) -> torch.Tensor:

        # x_t: B, T
        B = x_t.shape[0]
        T = x_t.shape[1]
        K = self.cfg.vocab_size
        
        if self.cfg.length_condition == "text":
            length_logits = self.length_predictor(
                text_feats, ~(text_mask.bool())
            )
        elif self.cfg.length_condition == "audio":
            length_logits = self.length_predictor(
                audio_feats, ~(audio_mask.bool())
            )
        else:
            raise ValueError(f"Unknown length_condition: {self.cfg.length_condition}")

        logits = self.dit(
            x_t, t,
            audio_feats, text_feats,
            ~(audio_mask.bool()), ~(text_mask.bool())
        )

        return logits, length_logits


if __name__ == "__main__":
    B = 2
    K = 650
    T_out = 16
    D = 1024
    n_H = 8

    cfg = DFMModelConfig(
        vocab_size=K,
        hidden_size=D,
        audio_dim=D,
        text_dim=D,
        num_heads=n_H
    )

    print(f"{cfg=}")
    model = DFMModel(cfg)

    x_t = torch.randint(0, K, (B, T_out))
    t = torch.rand(B)

    audio_feats = torch.rand((B, T_out * 4, D))
    audio_mask = torch.ones(B, T_out * 4).bool()

    text_feats = torch.rand((B, T_out * 2, D))
    text_mask = torch.ones(B, T_out * 2).bool()

    logits, length_logits = model(
        x_t, t,
        audio_feats, audio_mask,
        text_feats, text_mask
    )

    print(f"input audio: {audio_feats.shape}")
    print(f"input text: {text_feats.shape}")
    print(f"logits: {logits.shape}")
    print(f"length_logits: {length_logits.shape}")

    # test model wrapper
    wrapper = DFMModelWrapper(model)
    pred_lengths = wrapper.get_length_logits(text_feats, text_mask)
    print(f"pred_lengths: {pred_lengths.shape}")


