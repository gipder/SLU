import torch
import torch.nn as nn
from typing import Optional, Tuple

from fused_transformer import FusedTransformer


class FusedDecoderOnlyTransformer(FusedTransformer):
    """
    Fused Decoder-Only Transformer for SLU.

    A variant of FusedTransformer that removes the encoder step.
    The fused audio-text representation is fed directly into the decoder
    as cross-attention memory, without first passing through a TransformerEncoder.

    FusedTransformer pipeline:
        fusion → encoder(emb_fused) → memory → decoder(cross-attn to memory)

    FusedDecoderOnlyTransformer pipeline:
        fusion → emb_fused → decoder(cross-attn to emb_fused)   [encoder skipped]

    Everything else (fusion module, decoder, KV-cache decoding) is identical to
    FusedTransformer and inherited unchanged.

    Inputs
    ------
    input_ids   : (B, T)            target token ids
    audio_feats : (B, A, audio_dim) audio frame features
    text_feats  : (B, S, text_dim)  text (ASR hypothesis) features
    audio_mask  : (B, A)            True for padded audio positions
    text_mask   : (B, S)            True for padded text positions
    """

    # ── Override _encode: fuse then return directly (skip self.encoder) ────

    def _encode(
        self,
        audio_feats: Optional[torch.Tensor],
        text_feats:  Optional[torch.Tensor],
        audio_mask:  Optional[torch.Tensor],
        text_mask:   Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse audio into text via cross-attention; return fused directly as memory."""

        # Single-modality fallback: project and return without encoder
        if audio_feats is None and text_feats is None:
            raise ValueError("At least one of audio_feats or text_feats must be provided.")

        if audio_feats is None:
            return self.text_proj(text_feats) + self.modality_emb[1], text_mask

        if text_feats is None:
            return self.audio_proj(audio_feats) + self.modality_emb[0], audio_mask

        emb_aud  = self.audio_proj(audio_feats) + self.modality_emb[0]  # (B, A, H)
        emb_text = self.text_proj(text_feats)   + self.modality_emb[1]  # (B, S, H)

        emb_attn, _ = self.fusion_attn(
            emb_text, emb_aud, emb_aud,
            key_padding_mask=audio_mask,
            need_weights=False,
        )

        emb_fused = self.fusion_norm(self.fusion_proj(torch.cat([emb_text, emb_attn], dim=-1)))

        # Return fused directly — encoder is intentionally skipped
        return emb_fused, text_mask


if __name__ == "__main__":
    import time
    torch.manual_seed(0)

    B, A, S, D = 2, 100, 12, 1024
    K, DEPTH, N_HEAD = 50, 4, 4
    MAX_LEN = 32

    model = FusedDecoderOnlyTransformer(
        vocab_size=K, hidden_size=D, depth=DEPTH, num_heads=N_HEAD,
        audio_dim=D, text_dim=D, max_output_length=MAX_LEN,
    )
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    audio_feats = torch.randn(B, A, D)
    text_feats  = torch.randn(B, S, D)
    audio_mask  = torch.zeros(B, A, dtype=torch.bool)
    text_mask   = torch.zeros(B, S, dtype=torch.bool)
    input_ids   = torch.randint(0, K, (B, 8))

    logits = model(input_ids, audio_feats, text_feats, audio_mask, text_mask)
    print(f"logits: {logits.shape}")

    for use_cache in [False, True]:
        t0 = time.time()
        out = model.decode(audio_feats, text_feats, audio_mask, text_mask,
                           max_output_length=16, sos_id=1, eos_id=2, use_cache=use_cache)
        print(f"use_cache={use_cache}  shape={out.shape}  time={time.time()-t0:.3f}s")

    out_no_cache   = model.decode(audio_feats, text_feats, audio_mask, text_mask,
                                  max_output_length=16, use_cache=False)
    out_with_cache = model.decode(audio_feats, text_feats, audio_mask, text_mask,
                                  max_output_length=16, use_cache=True)
    print(f"Outputs match: {(out_no_cache == out_with_cache).all().item()}")

    # Single-modality fallback (text only)
    text_only = model.decode(None, text_feats, None, text_mask,
                             max_output_length=16, sos_id=1, eos_id=2)
    print(f"text-only decode shape: {text_only.shape}")
