from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    enabled: bool = False
    augment_type: str = "none"
    audio_noise_std: float = 0.02   # HuBERT-large: per-dim std ~0.5~1.5 → noise 0.01~0.05 권장
    text_noise_std: float = 0.01    # DeBERTa: 의미 밀도 높아 보수적으로 0.005~0.02 권장
    noise_schedule: str = "constant"  # constant | linear_increase | linear_decrease
    schedule_steps: int = 0           # linear_increase 시 warmup_step과 맞추면 안전
    use_layernorm: bool = False
    audio_mask_ratio: float = 0.1
    text_mask_ratio: float = 0.1
    audio_mask_span: int = 10
    text_mask_span: int = 3


class BaseAugmentor:
    def __init__(self, config: AugmentConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "none"

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
    @property
    def name(self) -> str:
        return "gaussian_noise"

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
        if not self.config.enabled:
            return audio_feats, text_feats, audio_mask, text_mask

        factor = self._schedule_factor(step)
        audio_std = self.config.audio_noise_std * factor
        text_std  = self.config.text_noise_std  * factor
        if audio_std <= 0 and text_std <= 0:
            return audio_feats, text_feats, audio_mask, text_mask

        #TODO: layer_norm 이 모델에 상관없이 augmentor에 있는게 이상함. default로 False
        if self.config.use_layernorm:
            audio_feats = F.layer_norm(audio_feats, (audio_feats.size(-1),))
            text_feats = F.layer_norm(text_feats, (text_feats.size(-1),))

        if audio_std > 0:
            audio_feats = audio_feats + self._masked_noise(audio_feats, audio_mask, audio_std)
        if text_std > 0:
            text_feats = text_feats + self._masked_noise(text_feats, text_mask, text_std)

        return audio_feats, text_feats, audio_mask, text_mask


class SpanMaskAugmentor(BaseAugmentor):
    @property
    def name(self) -> str:
        return "span_mask"

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


class CompositeAugmentor(BaseAugmentor):
    """여러 augmentor를 순서대로 적용합니다."""
    def __init__(self, augmentors: list) -> None:
        self.augmentors = augmentors

    @property
    def name(self) -> str:
        # 디렉토리명에 사용하기 안전한 구분자(-) 사용
        return "-".join(a.name for a in self.augmentors)

    def apply(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        for aug in self.augmentors:
            audio_feats, text_feats, audio_mask, text_mask = aug.apply(
                audio_feats, text_feats, audio_mask, text_mask, step
            )
        return audio_feats, text_feats, audio_mask, text_mask


def build_augmentor(args) -> BaseAugmentor:
    config = AugmentConfig(
        enabled=getattr(args, "use_augment", False),
        augment_type=getattr(args, "augment_type", "none"),
        audio_noise_std=getattr(args, "augment_audio_noise_std", 0.02),
        text_noise_std=getattr(args, "augment_text_noise_std", 0.01),
        noise_schedule=getattr(args, "augment_noise_schedule", "constant"),
        schedule_steps=getattr(args, "augment_noise_schedule_steps", 0),
        use_layernorm=getattr(args, "augment_use_layernorm", True),
        audio_mask_ratio=getattr(args, "augment_audio_mask_ratio", 0.1),
        text_mask_ratio=getattr(args, "augment_text_mask_ratio", 0.1),
        audio_mask_span=getattr(args, "augment_audio_mask_span", 10),
        text_mask_span=getattr(args, "augment_text_mask_span", 3),
    )

    if not config.enabled:
        return BaseAugmentor(config)

    # 쉼표 또는 + 구분으로 여러 타입 지정 가능: e.g. "span_mask,gaussian_noise" or "span_mask+gaussian_noise"
    raw = config.augment_type.replace("+", ",")
    type_list = [t.strip() for t in raw.split(",") if t.strip()]
    valid_types = [t for t in type_list if t in AUGMENTOR_REGISTRY and t != "none"]

    if len(valid_types) == 0:
        return BaseAugmentor(config)
    elif len(valid_types) == 1:
        return AUGMENTOR_REGISTRY[valid_types[0]](config)
    else:
        augmentors = [AUGMENTOR_REGISTRY[t](config) for t in valid_types]
        return CompositeAugmentor(augmentors)


def _print_feat_stats(label: str, feats: torch.Tensor, mask: torch.Tensor) -> None:
    valid = feats[mask.bool()]  # (N_valid, D)
    zero_frames = int((feats.abs().sum(-1) == 0).sum().item())
    print(f"  [{label}] mean={valid.mean():.4f}, std={valid.std():.4f}, "
          f"zero_frames={zero_frames}/{feats.size(0)*feats.size(1)}, "
          f"masked_positions={int((~mask.bool()).sum().item())}/{mask.numel()}")


if __name__ == "__main__":
    #torch.manual_seed(0)
    batch, t_audio, t_text, d = 2, 32, 10, 4
    audio_feats_orig = torch.randn(batch, t_audio, d)
    text_feats_orig  = torch.randn(batch, t_text, d)
    audio_mask_orig  = torch.ones(batch, t_audio, dtype=torch.bool)
    text_mask_orig   = torch.ones(batch, t_text, dtype=torch.bool)

    base_cfg = dict(
        enabled=True,
        audio_noise_std=0.02,
        text_noise_std=0.01,
        audio_mask_ratio=0.1,
        text_mask_ratio=0.05,
        audio_mask_span=5,
        text_mask_span=3,
    )

    cases = [
        ("Original (no aug)",        "none"),
        ("span_mask only",           "span_mask"),
        ("gaussian_noise only",      "gaussian_noise"),
        ("span_mask + gaussian_noise", "span_mask,gaussian_noise"),
    ]

    print("=" * 60)
    for title, aug_type in cases:
        cfg = AugmentConfig(**base_cfg, augment_type=aug_type)
        aug = build_augmentor(type("Args", (), {
            "use_augment": cfg.enabled,
            "augment_type": aug_type,
            "augment_audio_noise_std": cfg.audio_noise_std,
            "augment_text_noise_std": cfg.text_noise_std,
            "augment_noise_schedule": cfg.noise_schedule,
            "augment_noise_schedule_steps": cfg.schedule_steps,
            "augment_use_layernorm": cfg.use_layernorm,
            "augment_audio_mask_ratio": cfg.audio_mask_ratio,
            "augment_text_mask_ratio": cfg.text_mask_ratio,
            "augment_audio_mask_span": cfg.audio_mask_span,
            "augment_text_mask_span": cfg.text_mask_span,
        })())

        a, t, am, tm = aug.apply(
            audio_feats_orig.clone(), text_feats_orig.clone(),
            audio_mask_orig.clone(), text_mask_orig.clone(),
        )
        print(f"\n[{title}]  augmentor.name='{aug.name}'")
        _print_feat_stats("audio", a, am)
        _print_feat_stats("text ", t, tm)
    print("=" * 60)

    # ---- randomness 검증: span_mask를 5번 반복해서 positions가 다른지 확인 ----
    print("\n[Randomness check: span_mask × 5 runs]")
    cfg = AugmentConfig(**base_cfg, augment_type="span_mask")
    aug = build_augmentor(type("Args", (), {
        "use_augment": cfg.enabled,
        "augment_type": "span_mask",
        "augment_audio_noise_std": cfg.audio_noise_std,
        "augment_text_noise_std": cfg.text_noise_std,
        "augment_noise_schedule": cfg.noise_schedule,
        "augment_noise_schedule_steps": cfg.schedule_steps,
        "augment_use_layernorm": cfg.use_layernorm,
        "augment_audio_mask_ratio": cfg.audio_mask_ratio,
        "augment_text_mask_ratio": cfg.text_mask_ratio,
        "augment_audio_mask_span": cfg.audio_mask_span,
        "augment_text_mask_span": cfg.text_mask_span,
    })())
    masks_seen = []
    for i in range(5):
        _, _, am, _ = aug.apply(
            audio_feats_orig.clone(), text_feats_orig.clone(),
            audio_mask_orig.clone(), text_mask_orig.clone(),
        )
        masked_positions = (~am[0].bool()).nonzero(as_tuple=True)[0].tolist()
        print(f"  run {i+1}: audio[0] masked positions = {masked_positions}")
        masks_seen.append(masked_positions)
    all_same = all(m == masks_seen[0] for m in masks_seen)
    print(f"  → All runs identical? {all_same}  (should be False for proper randomness)")
