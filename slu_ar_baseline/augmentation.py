from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    enabled: bool = False
    augment_type: str = "none"
    noise_std: float = 0.0
    noise_schedule: str = "constant"  # constant | linear_increase | linear_decrease
    schedule_steps: int = 0
    use_layernorm: bool = False
    audio_mask_ratio: float = 0.1
    text_mask_ratio: float = 0.1
    audio_mask_span: int = 10
    text_mask_span: int = 3


class BaseAugmentor:
    def __init__(self, config: AugmentConfig) -> None:
        self.config = config

    def apply(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return audio_feats, text_feats, audio_mask, text_mask


class GaussianNoiseAugmentor(BaseAugmentor):
    def _schedule_factor(self, step: Optional[int]) -> float:
        if step is None or self.config.schedule_steps <= 0:
            return 1.0

        progress = min(1.0, max(0.0, step / float(self.config.schedule_steps)))
        if self.config.noise_schedule == "linear_increase":
            return progress
        if self.config.noise_schedule == "linear_decrease":
            return 1.0 - progress
        return 1.0

    def _masked_noise(self, x: torch.Tensor, mask: Optional[torch.Tensor], std: float) -> torch.Tensor:
        noise = torch.randn_like(x) * std
        if mask is None:
            return noise
        return noise * mask.unsqueeze(-1).to(noise.dtype)

    def apply(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.enabled or self.config.noise_std <= 0:
            return audio_feats, text_feats, audio_mask, text_mask

        factor = self._schedule_factor(step)
        std = self.config.noise_std * factor
        if std <= 0:
            return audio_feats, text_feats, audio_mask, text_mask

        if self.config.use_layernorm:
            audio_feats = F.layer_norm(audio_feats, (audio_feats.size(-1),))
            text_feats = F.layer_norm(text_feats, (text_feats.size(-1),))

        audio_feats = audio_feats + self._masked_noise(audio_feats, audio_mask, std)
        text_feats = text_feats + self._masked_noise(text_feats, text_mask, std)

        return audio_feats, text_feats, audio_mask, text_mask


class SpanMaskAugmentor(BaseAugmentor):
    def _apply_span_mask(
        self,
        feats: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_ratio: float,
        span_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask_ratio <= 0:
            return feats, mask
        if span_len <= 0:
            span_len = 1

        batch_size, time_steps, _ = feats.size()
        if mask is None:
            mask = torch.ones((batch_size, time_steps), device=feats.device, dtype=torch.bool)
        else:
            mask = mask.to(torch.bool)

        updated_mask = mask.clone()
        for b in range(batch_size):
            valid_len = int(mask[b].sum().item())
            if valid_len <= 0:
                continue
            total_to_mask = int(valid_len * mask_ratio)
            if total_to_mask <= 0:
                continue

            masked = 0
            max_start = max(1, valid_len - span_len + 1)
            while masked < total_to_mask:
                start = torch.randint(0, max_start, (1,), device=feats.device).item()
                end = min(valid_len, start + span_len)
                span_indices = torch.arange(start, end, device=feats.device)
                updated_mask[b, span_indices] = False
                masked += (end - start)

        feats = feats * updated_mask.unsqueeze(-1).to(feats.dtype)
        return feats, updated_mask

    def apply(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.enabled:
            return audio_feats, text_feats, audio_mask, text_mask

        audio_feats, audio_mask = self._apply_span_mask(
            feats=audio_feats,
            mask=audio_mask,
            mask_ratio=self.config.audio_mask_ratio,
            span_len=self.config.audio_mask_span,
        )
        text_feats, text_mask = self._apply_span_mask(
            feats=text_feats,
            mask=text_mask,
            mask_ratio=self.config.text_mask_ratio,
            span_len=self.config.text_mask_span,
        )

        return audio_feats, text_feats, audio_mask, text_mask


AUGMENTOR_REGISTRY = {
    "none": BaseAugmentor,
    "gaussian_noise": GaussianNoiseAugmentor,
    "span_mask": SpanMaskAugmentor,
    # Future: "token_dropout", "modality_dropout", etc.
}


def build_augmentor(args) -> BaseAugmentor:
    config = AugmentConfig(
        enabled=getattr(args, "augment", False),
        augment_type=getattr(args, "augment_type", "none"),
        noise_std=getattr(args, "augment_noise_std", 0.0),
        noise_schedule=getattr(args, "augment_noise_schedule", "constant"),
        schedule_steps=getattr(args, "augment_noise_schedule_steps", 0),
        use_layernorm=getattr(args, "augment_use_layernorm", True),
        audio_mask_ratio=getattr(args, "augment_audio_mask_ratio", 0.0),
        text_mask_ratio=getattr(args, "augment_text_mask_ratio", 0.0),
        audio_mask_span=getattr(args, "augment_audio_mask_span", 1),
        text_mask_span=getattr(args, "augment_text_mask_span", 1),
    )

    augmentor_cls = AUGMENTOR_REGISTRY.get(config.augment_type, BaseAugmentor)
    if config.augment_type not in AUGMENTOR_REGISTRY:
        config.enabled = False
    return augmentor_cls(config)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, t_audio, t_text, d = 2, 12, 8, 4
    audio_feats = torch.randn(batch, t_audio, d)
    text_feats = torch.randn(batch, t_text, d)
    audio_mask = torch.ones(batch, t_audio, dtype=torch.bool)
    text_mask = torch.ones(batch, t_text, dtype=torch.bool)

    cfg = AugmentConfig(
        enabled=True,
        augment_type="span_mask",
        audio_mask_ratio=0.3,
        text_mask_ratio=0.5,
        audio_mask_span=3,
        text_mask_span=2,
    )
    aug = SpanMaskAugmentor(cfg)    
    a2, t2, am2, tm2 = aug.apply(audio_feats, text_feats, audio_mask, text_mask)
    print("Original audio mask:", audio_mask)
    print("Original text mask:", text_mask)
    print("Augmented audio mask:", am2)
    print("Augmented text mask:", tm2)

    print("Audio masked frames:", int((~am2).sum().item()))
    print("Text masked frames:", int((~tm2).sum().item()))
    print("Audio mask ratio:", float((~am2).sum().item()) / float(am2.numel()))
    print("Text mask ratio:", float((~tm2).sum().item()) / float(tm2.numel()))
