import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

#from dit import DiscreteDualDiT
from basic_transformer import BasicTransformer
#from length_predictor import MaskedLengthPredictionModule
from flow_matching.utils import ModelWrapper

@dataclass
class ARModelConfig:
    # DIT 설정
    vocab_size: int = 42
    hidden_size: int = 512
    depth: int = 6
    num_heads: int = 8
    audio_dim: int = 1024
    text_dim: int = 1024
    max_output_length: int = 256    
    sos_token_id: int = 1
    eos_token_id: int = 2



class ARModel(nn.Module):
    def __init__(self, cfg: ARModelConfig):
        super().__init__()
        self.cfg = cfg        
        self.basic_transformer = BasicTransformer(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            audio_dim=cfg.audio_dim,
            text_dim=cfg.text_dim,
            max_output_length=cfg.max_output_length
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # x_t: B, T
        B = input_ids.shape[0]
        T = input_ids.shape[1]
        K = self.cfg.vocab_size

        # audio mask and test mask: True for audio/text embedding positions
        # convert to False for padding positions
        logits = self.basic_transformer(
            input_ids,
            audio_feats, text_feats,
            ~(audio_mask.bool()), ~(text_mask.bool())
        )

        return logits
    
    def decode(
        self,
        audio_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        max_output_length: Optional[int] = None,
        sos_id: int = 1,
        eos_id: Optional[int] = 2,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:        
        generated = self.basic_transformer.decode(
            audio_feats=audio_feats,
            text_feats=text_feats,
            audio_mask=~(audio_mask.bool()),
            text_mask=~(text_mask.bool()),
            max_output_length=max_output_length,
            sos_id=sos_id,
            eos_id=eos_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )

        return generated


if __name__ == "__main__":
    B = 2
    K = 650
    T_out = 16
    D = 1024
    n_H = 8

    cfg = ARModelConfig(
        vocab_size=K,
        hidden_size=D,
        audio_dim=D,
        text_dim=D,
        num_heads=n_H
    )

    print(f"{cfg=}")
    model = ARModel(cfg)

    input_ids = torch.randint(0, K, (B, T_out))

    audio_feats = torch.rand((B, T_out * 4, D))
    audio_mask = torch.ones(B, T_out * 4).bool()

    text_feats = torch.rand((B, T_out * 2, D))
    text_mask = torch.ones(B, T_out * 2).bool()

    logits = model(
        input_ids,
        audio_feats, 
        text_feats, 
        audio_mask,
        text_mask
    )

    print(f"input audio: {audio_feats.shape}")
    print(f"input text: {text_feats.shape}")
    print(f"logits: {logits.shape}")    

    #decode
    generated = model.decode(
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
        max_output_length=16,
        sos_id=1,
        eos_id=2,
        do_sample=False,
    )

    print(f"generated: {generated.shape}")

