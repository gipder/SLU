import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from dit import DiscreteDualDiT
from basic_transformer import BasicTransformer
from length_predictor import MaskedLengthPredictionModule
from flow_matching.utils import ModelWrapper

# for DFM testing
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver

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
    model_type: str = "transformer"  # "dit" or "transformer"
    norm_first: bool = True  # Whether to apply layer normalization before attention and FFN


class DFMModelWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        audio_feats = extras["audio_feats"]
        audio_mask = extras["audio_mask"]
        text_feats = extras["text_feats"]
        text_mask = extras["text_mask"]
        logits = self.model(x, t, audio_feats, audio_mask, text_feats, text_mask) # B, T_out, K
        prob = torch.nn.functional.softmax(logits.float(), dim=-1)
        return prob

    def predict_lengths(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor: # B
        predict_lengths = self.model.predict_lengths(x, x_mask)
        return predict_lengths


class ARModel(nn.Module):
    def __init__(self, cfg: ARModelConfig):
        super().__init__()
        self.cfg = cfg
        self.dit = None
        self.basic_transformer = None
        self.dfm_model = None
        if cfg.model_type == "dit":
            # Not implemented yet, but we can easily add DIT as an alternative to the basic transformer
            self.dit = DiscreteDualDiT(
                vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            audio_dim=cfg.audio_dim,
            text_dim=cfg.text_dim,
            )
            self.dfm_model = self.dit
        elif cfg.model_type == "transformer":
            self.basic_transformer = BasicTransformer(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                audio_dim=cfg.audio_dim,
                text_dim=cfg.text_dim,
                max_output_length=cfg.max_output_length
            )
            self.slu_model = self.basic_transformer
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")

        """
        self.length_predictor = MaskedLengthPredictionModule(
            embed_dim=cfg.embed_dim,
            length_hidden_dim=cfg.length_hidden_dim,
            max_target_positions=cfg.max_target_positions,
            length_dropout=cfg.length_dropout
        )
        """

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        text_feats: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
                # x_t: B, T
        B = input_ids.shape[0]
        T = input_ids.shape[1]
        K = self.cfg.vocab_size
        
        logits = self.slu_model(
            input_ids,
            audio_feats, text_feats,
            ~(audio_mask.bool()), ~(text_mask.bool())
        )

        return logits

    @torch.no_grad()
    def decode(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        max_output_length: Optional[int] = None,
        sos_id: int = 1,
        eos_id: Optional[int] = 2,
        use_cache: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.slu_model.decode(
            audio_feats, text_feats, ~(audio_mask.bool()), ~(text_mask.bool()),
            max_output_length, sos_id, eos_id, use_cache=use_cache, device=device
        )               

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
        num_heads=n_H,
        model_type="transformer",
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
        audio_feats, audio_mask,
        text_feats, text_mask
    )

    print(f"input audio: {audio_feats.shape}")
    print(f"input text: {text_feats.shape}")
    print(f"logits: {logits.shape}")

    # decoding test
    decoded_ids = model.decode(
        audio_feats, text_feats, audio_mask, text_mask,
        max_output_length=T_out, sos_id=1, eos_id=2,
        do_sample=False, temperature=1.0, top_k=None
    )

    print(f"decoded_ids: {decoded_ids.shape}")
    #print(f"length_logits: {length_logits.shape}")
   