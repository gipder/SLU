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

    Copy mechanism (enabled by use_copy=True):
        The copy mechanism allows the decoder to copy tokens directly from the
        ASR hypothesis.  At each decode step t:

            g_t = softmax(linear(d_t))            generative distribution
            r_t, w_t = MHA(d_t, e, e)            copy cross-attention
            c_t = Scatter(h, w_t)                 copy distribution over vocab
            P_copy = σ(Linear([d_t, r_t]))        copy gate
            o_t = (1-P_copy)*g_t + P_copy*c_t    final distribution

        Here e = token_emb(copy_ids) are embeddings of the ASR hypothesis
        tokens (HuBERT CTC vocabulary), and h = copy_ids are the token IDs.
        Using the decoder's own token_emb keeps the vocabulary aligned with
        the SLU output vocabulary.

    Inputs
    ------
    input_ids   : (B, T)            target token ids
    audio_feats : (B, A, audio_dim) audio frame features
    text_feats  : (B, S, text_dim)  text (ASR hypothesis) features
    audio_mask  : (B, A)            True for padded audio positions
    text_mask   : (B, S)            True for padded text positions
    copy_ids    : (B, C)            ASR hypothesis token ids (HuBERT CTC vocab)
    copy_mask   : (B, C)            True for padded copy positions
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
        use_copy: bool = False,
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

        self.use_copy = use_copy
        self._vocab_size = vocab_size

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

        # ── Copy mechanism ──────────────────────────────────────────────────
        if use_copy:
            # Separate cross-attention over ASR hypothesis token embeddings
            self.copy_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            # Copy gate: σ(Linear([d_t, r_t]))
            self.copy_gate = nn.Linear(hidden_size * 2, 1)
            nn.init.xavier_uniform_(self.copy_gate.weight)
            nn.init.zeros_(self.copy_gate.bias)

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
            if text_feats is None and audio_feats is not None:
                # Audio-only: subsample by 1/4 to avoid OOM on long sequences
                audio_feats = audio_feats[:, ::4, :]
                if audio_mask is not None:
                    audio_mask = audio_mask[:, ::4]
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

    # ── Copy mechanism helpers ──────────────────────────────────────────────

    def _apply_copy(
        self,
        d:        torch.Tensor,
        copy_src: torch.Tensor,
        copy_ids: torch.Tensor,
        copy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply the copy mechanism and return the final mixed distribution o_t.

        Args:
            d        : (B, T, H)   decoder hidden states
            copy_src : (B, C, H)   copy source embeddings = token_emb(copy_ids)
            copy_ids : (B, C)      ASR hypothesis token ids
            copy_mask: (B, C)      True for padded copy positions
        Returns:
            o        : (B, T, V)   probability distribution over vocabulary
        """
        # Cast to float32 for numerical stability (prevents negative NLLLoss under AMP fp16)
        d32        = d.float()
        copy_src32 = copy_src.float()

        B, T, _ = d32.shape
        C = copy_ids.size(1)

        # g_t = softmax(linear(d_t))
        g = torch.softmax(self.head(d32), dim=-1)  # (B, T, V)  float32

        # r_t, w_t = MHA(d_t, e, e)
        r, w = self.copy_attn(
            d32, copy_src32, copy_src32,
            key_padding_mask=copy_mask,
            need_weights=True,
            average_attn_weights=True,
        )  # r: (B, T, H),  w: (B, T, C)  — float32 from MHA

        # Clamp to [0, 1] to guard against any residual fp precision drift
        w = w.float().clamp(min=0.0, max=1.0)

        # c_t = Scatter(h, w_t): sum attention weights by vocab token id
        c = torch.zeros(B, T, self._vocab_size, device=d.device, dtype=torch.float32)
        c.scatter_add_(
            dim=-1,
            index=copy_ids.unsqueeze(1).expand(B, T, C),
            src=w,
        )

        # P_copy = σ(Linear([d_t, r_t]))
        p_copy = torch.sigmoid(self.copy_gate(torch.cat([d32, r.float()], dim=-1)))  # (B, T, 1)

        # o_t = (1-P_copy) * g_t + P_copy * c_t
        o = (1.0 - p_copy) * g + p_copy * c  # (B, T, V)

        # Renormalize to ensure valid probability distribution (guards fp drift)
        o = o / o.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return o  # (B, T, V)  float32

    # ── Forward (teacher-forced training) — overrides parent ───────────────

    def forward(
        self,
        input_ids:   torch.Tensor,
        audio_feats: Optional[torch.Tensor] = None,
        text_feats:  Optional[torch.Tensor] = None,
        audio_mask:  Optional[torch.Tensor] = None,
        text_mask:   Optional[torch.Tensor] = None,
        copy_ids:    Optional[torch.Tensor] = None,
        copy_mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            If use_copy and copy_ids given: log-probabilities (B, T, V)
            Otherwise: logits (B, T, V)
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.pos_emb.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_output_length {self.pos_emb.size(1)}."
            )

        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        memory, memory_key_padding_mask = self._encode(
            audio_feats, text_feats, audio_mask, text_mask
        )
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        d = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        if self.use_copy and copy_ids is not None:
            copy_src = self.token_emb(copy_ids)  # (B, C, H)
            o = self._apply_copy(d, copy_src, copy_ids, copy_mask)  # float32, normalized
            return torch.log(o.clamp(min=1e-9))  # float32 log-probs → use NLLLoss

        return self.head(d)  # standard logits → use CrossEntropyLoss

    # ── Autoregressive decoding — overrides parent ─────────────────────────

    @torch.no_grad()
    def decode(
        self,
        audio_feats:       Optional[torch.Tensor] = None,
        text_feats:        Optional[torch.Tensor] = None,
        audio_mask:        Optional[torch.Tensor] = None,
        text_mask:         Optional[torch.Tensor] = None,
        max_output_length: Optional[int] = None,
        sos_id:            int = 1,
        eos_id:            int = 2,
        use_cache:         bool = True,
        device:            Optional[torch.device] = None,
        copy_ids:          Optional[torch.Tensor] = None,
        copy_mask:         Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if device is None:
            device = audio_feats.device if audio_feats is not None else text_feats.device

        max_output_length = max_output_length or self.pos_emb.size(1)

        # Encode conditioning once
        memory, memory_key_padding_mask = self._encode(
            audio_feats, text_feats, audio_mask, text_mask
        )

        batch_size = memory.size(0)
        generated  = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Pre-compute copy source embeddings (fixed across all steps)
        use_copy_now = self.use_copy and copy_ids is not None
        copy_src = self.token_emb(copy_ids) if use_copy_now else None  # (B, C, H)

        kv_cache = None
        if use_cache:
            head_dim     = self.hidden_size // self.num_heads
            cache_dtype  = self.token_emb.weight.dtype
            kv_cache = {
                'k': [torch.zeros(batch_size, 0, self.num_heads, head_dim,
                                  device=device, dtype=cache_dtype) for _ in range(self.depth)],
                'v': [torch.zeros(batch_size, 0, self.num_heads, head_dim,
                                  device=device, dtype=cache_dtype) for _ in range(self.depth)],
            }

        for _ in range(max_output_length - 1):
            seq_len = generated.size(1)
            if seq_len > self.pos_emb.size(1):
                break

            if use_cache:
                x = self.token_emb(generated[:, -1:]) + self.pos_emb[:, seq_len - 1 : seq_len, :]
                for layer_idx, layer in enumerate(self.decoder.layers):
                    residual = x
                    x_norm = layer.norm1(x) if self.norm_first else x

                    curr_k, curr_v = self._compute_kv(x_norm, layer.self_attn)
                    k_full = torch.cat([kv_cache['k'][layer_idx], curr_k], dim=1)
                    v_full = torch.cat([kv_cache['v'][layer_idx], curr_v], dim=1)
                    kv_cache['k'][layer_idx] = k_full
                    kv_cache['v'][layer_idx] = v_full

                    if self.norm_first:
                        x = residual + self._multihead_attention(
                            layer.norm1(x), k_full, v_full, layer.self_attn
                        )[0]
                        residual = x
                        x = residual + layer.multihead_attn(
                            layer.norm2(x), memory, memory,
                            key_padding_mask=memory_key_padding_mask, need_weights=False
                        )[0]
                        residual = x
                        x = residual + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
                    else:
                        x = layer.norm1(residual + self._multihead_attention(
                            x, k_full, v_full, layer.self_attn
                        )[0])
                        residual = x
                        x = layer.norm2(residual + layer.multihead_attn(
                            x, memory, memory,
                            key_padding_mask=memory_key_padding_mask, need_weights=False
                        )[0])
                        residual = x
                        x = layer.norm3(residual + layer.linear2(layer.activation(layer.linear1(x))))

                x = self.decoder.norm(x)
            else:
                x = self.token_emb(generated) + self.pos_emb[:, :seq_len, :]
                tgt_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
                )
                x = self.decoder(
                    x, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            # ── Token selection ───────────────────────────────────────────
            d_t = x[:, -1:, :]  # (B, 1, H)
            if use_copy_now:
                o = self._apply_copy(d_t, copy_src, copy_ids, copy_mask)  # (B, 1, V)
                next_token = torch.argmax(o.squeeze(1), dim=-1, keepdim=True)
            else:
                logits     = self.head(d_t.squeeze(1))                    # (B, V)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_token = torch.where(
                finished.unsqueeze(1), torch.tensor(eos_id, device=device), next_token
            )
            generated = torch.cat([generated, next_token], dim=1)
            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return generated


if __name__ == "__main__":
    import time
    torch.manual_seed(0)

    B, A, S, D = 2, 100, 12, 1024
    K, DEPTH, N_HEAD = 50, 4, 4
    MAX_LEN = 32
    C = 8  # ASR hypothesis length (HuBERT CTC tokens)

    # ── Without copy mechanism ─────────────────────────────────────────────
    print("=" * 60)
    print("Test 1: FusedTransformer WITHOUT copy mechanism")
    print("=" * 60)
    model_no_copy = FusedTransformer(
        vocab_size=K, hidden_size=D, depth=DEPTH, num_heads=N_HEAD,
        audio_dim=D, text_dim=D, max_output_length=MAX_LEN,
        use_copy=False,
    )
    model_no_copy.eval()
    print(f"Parameters: {sum(p.numel() for p in model_no_copy.parameters()):,}")

    audio_feats = torch.randn(B, A, D)
    text_feats  = torch.randn(B, S, D)
    audio_mask  = torch.zeros(B, A, dtype=torch.bool)
    text_mask   = torch.zeros(B, S, dtype=torch.bool)
    input_ids   = torch.randint(0, K, (B, 8))

    logits = model_no_copy(input_ids, audio_feats, text_feats, audio_mask, text_mask)
    print(f"logits: {logits.shape}")

    for use_cache in [False, True]:
        t0 = time.time()
        out = model_no_copy.decode(audio_feats, text_feats, audio_mask, text_mask,
                                   max_output_length=16, sos_id=1, eos_id=2, use_cache=use_cache)
        print(f"use_cache={use_cache}  shape={out.shape}  time={time.time()-t0:.3f}s")

    out_no_cache   = model_no_copy.decode(audio_feats, text_feats, audio_mask, text_mask,
                                          max_output_length=16, use_cache=False)
    out_with_cache = model_no_copy.decode(audio_feats, text_feats, audio_mask, text_mask,
                                          max_output_length=16, use_cache=True)
    print(f"Outputs match (no_copy): {(out_no_cache == out_with_cache).all().item()}")

    # Single-modality fallback (text only)
    text_only = model_no_copy.decode(None, text_feats, None, text_mask,
                                     max_output_length=16, sos_id=1, eos_id=2)
    print(f"text-only decode shape: {text_only.shape}")

    # ── With copy mechanism ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Test 2: FusedTransformer WITH copy mechanism")
    print("=" * 60)
    model_copy = FusedTransformer(
        vocab_size=K, hidden_size=D, depth=DEPTH, num_heads=N_HEAD,
        audio_dim=D, text_dim=D, max_output_length=MAX_LEN,
        use_copy=True,
    )
    model_copy.eval()
    print(f"Parameters: {sum(p.numel() for p in model_copy.parameters()):,}")

    # copy_ids: ASR hypothesis token IDs (HuBERT CTC vocab, length C)
    copy_ids  = torch.randint(0, K, (B, C))
    copy_mask = torch.zeros(B, C, dtype=torch.bool)  # no padding

    # Forward: returns log-probabilities
    log_probs = model_copy(input_ids, audio_feats, text_feats, audio_mask, text_mask,
                           copy_ids=copy_ids, copy_mask=copy_mask)
    print(f"log_probs: {log_probs.shape}")
    # Verify it is log-probs: exp should sum to ~1
    prob_sum = log_probs.exp().sum(dim=-1)
    print(f"prob_sum (should be ~1): min={prob_sum.min():.4f}  max={prob_sum.max():.4f}")

    # Decode with copy
    for use_cache in [False, True]:
        t0 = time.time()
        out = model_copy.decode(audio_feats, text_feats, audio_mask, text_mask,
                                max_output_length=16, sos_id=1, eos_id=2, use_cache=use_cache,
                                copy_ids=copy_ids, copy_mask=copy_mask)
        print(f"use_cache={use_cache}  shape={out.shape}  time={time.time()-t0:.3f}s")

    out_no_cache   = model_copy.decode(audio_feats, text_feats, audio_mask, text_mask,
                                       max_output_length=16, use_cache=False,
                                       copy_ids=copy_ids, copy_mask=copy_mask)
    out_with_cache = model_copy.decode(audio_feats, text_feats, audio_mask, text_mask,
                                       max_output_length=16, use_cache=True,
                                       copy_ids=copy_ids, copy_mask=copy_mask)
    print(f"Outputs match (copy): {(out_no_cache == out_with_cache).all().item()}")

    # Audio-only subsampling (1/4)
    print()
    print("=" * 60)
    print("Test 3: Audio-only fallback with 1/4 subsampling")
    print("=" * 60)
    A_long = 800  # long audio that would OOM without subsampling
    audio_long = torch.randn(B, A_long, D)
    audio_mask_long = torch.zeros(B, A_long, dtype=torch.bool)
    out_audio_only = model_no_copy.decode(
        audio_long, None, audio_mask_long, None,
        max_output_length=16, sos_id=1, eos_id=2, use_cache=True,
    )
    print(f"audio length: {A_long} → subsampled to {A_long // 4}")
    print(f"audio-only decode shape: {out_audio_only.shape}")

    # verify mask is also subsampled correctly (partial padding)
    audio_mask_partial = torch.zeros(B, A_long, dtype=torch.bool)
    audio_mask_partial[:, A_long // 2 :] = True  # second half is padding
    out_masked = model_no_copy.decode(
        audio_long, None, audio_mask_partial, None,
        max_output_length=16, sos_id=1, eos_id=2, use_cache=True,
    )
    print(f"audio-only (partial padding) decode shape: {out_masked.shape}")

    # Copy with padding in hypothesis
    print()
    print("=" * 60)
    print("Test 4: Copy with padded hypothesis")
    print("=" * 60)
    copy_ids_pad  = torch.randint(0, K, (B, C + 4))
    copy_mask_pad = torch.zeros(B, C + 4, dtype=torch.bool)
    copy_mask_pad[:, C:] = True  # last 4 positions are padding
    out_pad = model_copy.decode(audio_feats, text_feats, audio_mask, text_mask,
                                max_output_length=16, sos_id=1, eos_id=2, use_cache=True,
                                copy_ids=copy_ids_pad, copy_mask=copy_mask_pad)
    print(f"decode with padded copy_ids shape: {out_pad.shape}")

    print()
    print("All tests passed.")
