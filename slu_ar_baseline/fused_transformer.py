import torch
import torch.nn as nn
from typing import Optional, Tuple

from encoder_decoder_transformer import EncoderDecoderTransformer


class FusedTransformer(EncoderDecoderTransformer):
    """
    Fused Encoder-Decoder Transformer for SLU.

    A variant of EncoderDecoderTransformer where audio and text embeddings
    are combined via a fusion module before entering the encoder, rather than
    being simply concatenated.

    Fusion:
        emb_aud  = audio_proj(audio_feats) + modality_emb[0]   (B, A, H)
        emb_text = text_proj(text_feats)   + modality_emb[1]   (B, S, H)
        emb_attn = MHA(query=emb_text, key=emb_aud, value=emb_aud)
        emb_fused = Linear(cat([emb_text, emb_attn], dim=-1))   (B, S, H)

    emb_fused (text-length sequence) is then fed into the encoder.
    The decoder generates tokens autoregressively by cross-attending to
    the encoder output.

    Inputs
    ------
    input_ids   : (B, T)            target token ids
    audio_feats : (B, A, audio_dim) audio frame features
    text_feats  : (B, S, text_dim)  text (ASR hypothesis) features
    audio_mask  : (B, A)            True for padded audio positions
    text_mask   : (B, S)            True for padded text positions
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        audio_dim: int = 1024,
        text_dim: int = 1024,
        max_output_length: int = 256,
        dropout: float = 0.1,
        norm_first: bool = True,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            audio_dim=audio_dim,
            text_dim=text_dim,
            max_output_length=max_output_length,
            dropout=dropout,
            norm_first=norm_first,
        )

        # ── Fusion module ───────────────────────────────────────────────────
        # Cross-attention: text queries audio
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Project [emb_text || emb_attn] -> hidden_size
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_norm = nn.LayerNorm(hidden_size)

        nn.init.xavier_uniform_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

    # ── Override _encode to use fusion ─────────────────────────────────────

    def _encode(
        self,
        audio_feats: Optional[torch.Tensor],
        text_feats:  Optional[torch.Tensor],
        audio_mask:  Optional[torch.Tensor],
        text_mask:   Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse audio into text via cross-attention, then run encoder."""

        # Fall back to parent (concat) when only one modality is available
        if audio_feats is None or text_feats is None:
            return super()._encode(audio_feats, text_feats, audio_mask, text_mask)

        emb_aud  = self.audio_proj(audio_feats) + self.modality_emb[0]  # (B, A, H)
        emb_text = self.text_proj(text_feats)   + self.modality_emb[1]  # (B, S, H)

        # emb_attn = MHA(query=emb_text, key=emb_aud, value=emb_aud)
        emb_attn, _ = self.fusion_attn(
            emb_text, emb_aud, emb_aud,
            key_padding_mask=audio_mask,  # True = padded audio frame
            need_weights=False,
        )

        # emb_stack = Stack(emb_text, emb_attn) -> (B, S, 2H)
        emb_stack = torch.cat([emb_text, emb_attn], dim=-1)

        # emb_fused = Linear(emb_stack) -> (B, S, H)
        emb_fused = self.fusion_norm(self.fusion_proj(emb_stack))

        memory = self.encoder(emb_fused, src_key_padding_mask=text_mask)
        return memory, text_mask


if __name__ == "__main__":
    import time
    torch.manual_seed(0)

    B, A, S, D = 2, 100, 12, 1024
    K, DEPTH, N_HEAD = 50, 4, 4
    MAX_LEN = 32

    model = FusedTransformer(
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
