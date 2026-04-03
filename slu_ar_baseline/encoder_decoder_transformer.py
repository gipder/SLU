import torch
import torch.nn as nn
from typing import Optional, Tuple


class EncoderDecoderTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for SLU.

    The encoder refines the concatenated audio + text projections into a
    single memory sequence.  The decoder generates output tokens
    autoregressively by cross-attending to that encoder memory.

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
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth if depth % 2 == 0 else depth + 1  # Ensure even depth for equal encoder/decoder layers
        self.norm_first = norm_first

        # ── Input projections ──────────────────────────────────────────────
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.text_proj  = nn.Linear(text_dim,  hidden_size)
        self.modality_emb = nn.Parameter(torch.zeros(2, hidden_size))

        # ── Encoder (processes concatenated audio + text) ──────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth//2,
            norm=nn.LayerNorm(hidden_size),
        )

        # ── Decoder (autoregressive, cross-attends to encoder output) ──────
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb   = nn.Parameter(torch.zeros(1, max_output_length, hidden_size))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=depth//2,
            norm=nn.LayerNorm(hidden_size),
        )

        self.head = nn.Linear(hidden_size, vocab_size)

        # ── Initialisation (same policy as BasicTransformer) ───────────────
        nn.init.normal_(self.pos_emb,     std=0.02)
        nn.init.normal_(self.modality_emb, std=0.02)
        nn.init.xavier_uniform_(self.audio_proj.weight)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.audio_proj.bias)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _encode(
        self,
        audio_feats: Optional[torch.Tensor],
        text_feats:  Optional[torch.Tensor],
        audio_mask:  Optional[torch.Tensor],
        text_mask:   Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project, concatenate, and encode audio + text into encoder memory."""
        parts, masks = [], []

        if audio_feats is not None:
            parts.append(self.audio_proj(audio_feats) + self.modality_emb[0])
            if audio_mask is not None:
                masks.append(audio_mask)

        if text_feats is not None:
            parts.append(self.text_proj(text_feats) + self.modality_emb[1])
            if text_mask is not None:
                masks.append(text_mask)

        if not parts:
            raise ValueError("At least one of audio_feats or text_feats must be provided.")

        src = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        src_key_padding_mask = torch.cat(masks, dim=1) if masks else None

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def _compute_kv(self, x: torch.Tensor, attn_module: nn.MultiheadAttention):
        """Compute K, V projections from x (used for KV-cache)."""
        B, T, _ = x.shape
        head_dim = self.hidden_size // self.num_heads
        _, w_k, w_v = attn_module.in_proj_weight.chunk(3)
        b_k, b_v = (
            (attn_module.in_proj_bias.chunk(3)[1], attn_module.in_proj_bias.chunk(3)[2])
            if attn_module.in_proj_bias is not None else (None, None)
        )
        k = torch.nn.functional.linear(x, w_k, b_k).view(B, T, self.num_heads, head_dim)
        v = torch.nn.functional.linear(x, w_v, b_v).view(B, T, self.num_heads, head_dim)
        return k, v

    def _multihead_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_module: nn.MultiheadAttention,
        need_weights: bool = False,
    ):
        """Manual multihead attention using pre-computed K, V (for KV-cache)."""
        B, q_len, _ = q.shape
        _, kv_len, _, head_dim = k.shape

        w_q = attn_module.in_proj_weight.chunk(3)[0]
        b_q = attn_module.in_proj_bias.chunk(3)[0] if attn_module.in_proj_bias is not None else None
        q = torch.nn.functional.linear(q, w_q, b_q)
        q = q.view(B, q_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            weights = None
        else:
            scores  = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            weights = torch.nn.functional.softmax(scores, dim=-1)
            out     = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, q_len, self.hidden_size)
        out = torch.nn.functional.linear(out, attn_module.out_proj.weight, attn_module.out_proj.bias)
        return out, weights if need_weights else None

    # ── Forward (teacher-forced training) ─────────────────────────────────

    def forward(
        self,
        input_ids:   torch.Tensor,
        audio_feats: Optional[torch.Tensor] = None,
        text_feats:  Optional[torch.Tensor] = None,
        audio_mask:  Optional[torch.Tensor] = None,
        text_mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.head(x)

    # ── Autoregressive decoding ────────────────────────────────────────────

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
    ) -> torch.Tensor:
        if device is None:
            device = audio_feats.device if audio_feats is not None else text_feats.device

        max_output_length = max_output_length or self.pos_emb.size(1)

        # Encode conditioning once — reused at every decode step
        memory, memory_key_padding_mask = self._encode(
            audio_feats, text_feats, audio_mask, text_mask
        )

        batch_size = memory.size(0)
        generated  = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished   = torch.zeros(batch_size, dtype=torch.bool, device=device)

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

            logits     = self.head(x[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token = torch.where(finished.unsqueeze(1), torch.tensor(eos_id, device=device), next_token)
            generated  = torch.cat([generated, next_token], dim=1)
            finished  |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return generated


if __name__ == "__main__":
    import time
    torch.manual_seed(0)

    B, A, S, D = 2, 100, 12, 1024
    K, DEPTH, N_HEAD = 50, 4, 4
    MAX_LEN = 32

    model = EncoderDecoderTransformer(
        vocab_size=K, hidden_size=D, depth=DEPTH, num_heads=N_HEAD,
        audio_dim=D, text_dim=D, max_output_length=MAX_LEN,
    )
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
