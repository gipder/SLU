import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from basic_transformer import BasicTransformer
from encoder_decoder_transformer import EncoderDecoderTransformer
from fused_transformer import FusedTransformer
from fused_decoder_only_transformer import FusedDecoderOnlyTransformer

@dataclass
class ARModelConfig:
    vocab_size: int = 42
    hidden_size: int = 512
    depth: int = 6
    num_heads: int = 8
    audio_dim: int = 1024
    text_dim: int = 1024
    max_output_length: int = 256
    sos_token_id: int = 1
    eos_token_id: int = 2
    model_type: str = "transformer"  # "transformer", "encoder_decoder_transformer", "fused_transformer", or "fused_decoder_only_transformer"
    norm_first: bool = True
    use_copy: bool = False  # Enable copy mechanism (fused_transformer / fused_decoder_only_transformer only)


class ARModel(nn.Module):
    def __init__(self, cfg: ARModelConfig):
        super().__init__()
        self.cfg = cfg
        self.basic_transformer = None
        self.encoder_decoder_transformer = None

        _common = dict(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            audio_dim=cfg.audio_dim,
            text_dim=cfg.text_dim,
            max_output_length=cfg.max_output_length,
            norm_first=cfg.norm_first,
        )

        if cfg.model_type == "transformer":
            self.basic_transformer = BasicTransformer(**_common)
            self.slu_model = self.basic_transformer
        elif cfg.model_type == "encoder_decoder_transformer":
            self.encoder_decoder_transformer = EncoderDecoderTransformer(**_common)
            self.slu_model = self.encoder_decoder_transformer
        elif cfg.model_type == "fused_transformer":
            self.fused_transformer = FusedTransformer(**_common, use_copy=cfg.use_copy)
            self.slu_model = self.fused_transformer
        elif cfg.model_type == "fused_decoder_only_transformer":
            self.fused_decoder_only_transformer = FusedDecoderOnlyTransformer(**_common, use_copy=cfg.use_copy)
            self.slu_model = self.fused_decoder_only_transformer
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
        audio_feats: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        text_feats: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        copy_ids: Optional[torch.Tensor] = None,
        copy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x_t: B, T
        B = input_ids.shape[0]
        T = input_ids.shape[1]
        K = self.cfg.vocab_size

        # copy kwargs forwarded only to models that support it
        copy_kwargs = {}
        if self.cfg.use_copy:
            copy_kwargs = {"copy_ids": copy_ids, "copy_mask": copy_mask}

        if audio_feats is None and audio_mask is None:
            logits = self.slu_model(
                input_ids,
                None, text_feats,
                None, ~(text_mask.bool()),
                **copy_kwargs
            )
        elif text_feats is None and text_mask is None:
            logits = self.slu_model(
                input_ids,
                audio_feats, None,
                ~(audio_mask.bool()), None,
                **copy_kwargs
            )
        else:
            logits = self.slu_model(
                input_ids,
                audio_feats, text_feats,
                ~(audio_mask.bool()), ~(text_mask.bool()),
                **copy_kwargs
            )
        return logits

    @torch.no_grad()
    def decode(
        self,
        audio_feats: torch.Tensor = None,
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
        copy_ids: Optional[torch.Tensor] = None,
        copy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        copy_kwargs = {}
        if self.cfg.use_copy:
            copy_kwargs = {"copy_ids": copy_ids, "copy_mask": copy_mask}

        if audio_feats is None and audio_mask is None:
            return self.slu_model.decode(
                None, text_feats, None, ~(text_mask.bool()),
                max_output_length, sos_id, eos_id, use_cache=use_cache, device=device,
                **copy_kwargs
            )
        elif text_feats is None and text_mask is None:
            return self.slu_model.decode(
                audio_feats, None, ~(audio_mask.bool()), None,
                max_output_length, sos_id, eos_id, use_cache=use_cache, device=device,
                **copy_kwargs
            )

        return self.slu_model.decode(
            audio_feats, text_feats, ~(audio_mask.bool()), ~(text_mask.bool()),
            max_output_length, sos_id, eos_id, use_cache=use_cache, device=device,
            **copy_kwargs
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
