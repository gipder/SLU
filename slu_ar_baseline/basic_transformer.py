import torch
import torch.nn as nn
from typing import Optional, Tuple


class BasicTransformer(nn.Module):
    """
    Autoregressive Transformer decoder with cross-attention to fused audio/text features.

    Inputs
    ------
    input_ids: (B, T) token ids
    audio_feats: (B, A, audio_dim)
    text_feats: (B, S, text_dim)
    audio_mask/text_mask: (B, A)/(B, S) with True for padded positions
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
        self.depth = depth
        self.norm_first = norm_first

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_output_length, hidden_size))

        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.modality_emb = nn.Parameter(torch.zeros(2, hidden_size))

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
            num_layers=depth,
            norm=nn.LayerNorm(hidden_size),
        )

        self.head = nn.Linear(hidden_size, vocab_size)
        
        # Proper initialization
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.modality_emb, std=0.02)
        
        # Initialize projection layers with smaller std for stability
        nn.init.xavier_uniform_(self.audio_proj.weight)
        nn.init.xavier_uniform_(self.text_proj.weight)
        if self.audio_proj.bias is not None:
            nn.init.zeros_(self.audio_proj.bias)
        if self.text_proj.bias is not None:
            nn.init.zeros_(self.text_proj.bias)
        
        # Initialize output head with small weights
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _build_memory(
        self,
        audio_feats: Optional[torch.Tensor],
        text_feats: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        memory_list = []
        mask_list = []

        if audio_feats is not None:
            audio = self.audio_proj(audio_feats) + self.modality_emb[0]
            memory_list.append(audio)
            if audio_mask is not None:
                mask_list.append(audio_mask)

        if text_feats is not None:
            text = self.text_proj(text_feats) + self.modality_emb[1]
            memory_list.append(text)
            if text_mask is not None:
                mask_list.append(text_mask)

        if not memory_list:
            raise ValueError("At least one of audio_feats or text_feats must be provided.")

        if len(memory_list) == 1:
            memory = memory_list[0]
            memory_key_padding_mask = mask_list[0] if mask_list else None
        else:
            memory = torch.cat(memory_list, dim=1)
            memory_key_padding_mask = torch.cat(mask_list, dim=1) if mask_list else None

        return memory, memory_key_padding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape

        if seq_len > self.pos_emb.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.pos_emb.size(1)}."
            )

        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        memory, memory_key_padding_mask = self._build_memory(
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
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def decode(
        self,
        audio_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        max_output_length: Optional[int] = None,
        sos_id: int = 1,
        eos_id: int = 2,
        use_cache: bool = True,
        #use_amp: bool = False,        
        device: Optional[torch.device] = None,        
    ) -> torch.Tensor:
        """
        Autoregressive decoding.

        Args:
            audio_feats/text_feats: conditioning features
            audio_mask/text_mask: key padding masks (True for padded positions)
            max_len: maximum output length (defaults to model max_seq_len)
            start_token_id: BOS token id
            end_token_id: EOS token id (optional)
            do_sample: if True, sample from distribution instead of greedy
            temperature: sampling temperature
            top_k: if set, restrict sampling to top-k logits
            device: device to run decoding on

        Returns:
            Tensor of shape (B, T) with generated token ids
        """
        if device is None:
            if audio_feats is not None:
                device = audio_feats.device
            elif text_feats is not None:
                device = text_feats.device
            else:
                raise ValueError("At least one of audio_feats or text_feats must be provided.")

        max_output_length = max_output_length or self.pos_emb.size(1)
        memory, memory_key_padding_mask = self._build_memory(
            audio_feats, text_feats, audio_mask, text_mask
        )

        batch_size = memory.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=sos_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        kv_cache = None
        if use_cache:
            # Transformer KV 캐시 초기화
            # 각 layer에서 self-attention K/V를 저장
            head_dim = self.hidden_size // self.num_heads
            cache_dtype = self.token_emb.weight.dtype
            kv_cache = {
                'k': [torch.zeros(batch_size, 0, self.num_heads, head_dim, device=device, dtype=cache_dtype)
                        for _ in range(self.depth)],
                'v': [torch.zeros(batch_size, 0, self.num_heads, head_dim, device=device, dtype=cache_dtype)
                        for _ in range(self.depth)],
            }
        """
        autocast_ctx = torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda" and not self.use_mamba,
        )
        """        
        for _ in range(max_output_length - 1):
            seq_len = generated.size(1)
            if seq_len > self.pos_emb.size(1):
                break

            if use_cache:
                # 핵심: 마지막 토큰 1개만 임베딩 (incremental decoding)
                x = self.token_emb(generated[:, -1:]) + self.pos_emb[:, seq_len - 1 : seq_len, :]
            else:
                x = self.token_emb(generated) + self.pos_emb[:, :seq_len, :]

            if use_cache:
                for layer_idx, layer in enumerate(self.decoder.layers):
                    residual = x

                    if self.norm_first:
                        # Pre-LN: norm → attn → residual
                        curr_k, curr_v = self._compute_kv(layer.norm1(x), layer.self_attn)
                    else:
                        # Post-LN: attn → residual → norm
                        curr_k, curr_v = self._compute_kv(x, layer.self_attn)

                    # 캐시에 새 토큰의 K, V만 추가
                    prev_k = kv_cache['k'][layer_idx]
                    prev_v = kv_cache['v'][layer_idx]
                    k_full = torch.cat([prev_k, curr_k], dim=1) if prev_k.shape[1] > 0 else curr_k
                    v_full = torch.cat([prev_v, curr_v], dim=1) if prev_v.shape[1] > 0 else curr_v
                    kv_cache['k'][layer_idx] = k_full
                    kv_cache['v'][layer_idx] = v_full

                    if self.norm_first:
                        # Self-attention (Pre-LN)
                        self_attn_out = self._multihead_attention(
                            layer.norm1(x), k_full, v_full, layer.self_attn, need_weights=False
                        )[0]
                        x = residual + self_attn_out

                        # Cross-attention (Pre-LN)
                        residual = x
                        cross_attn_out = layer.multihead_attn(
                            layer.norm2(x), memory, memory,
                            key_padding_mask=memory_key_padding_mask, need_weights=False
                        )[0]
                        x = residual + cross_attn_out

                        # Feed-forward (Pre-LN)
                        residual = x
                        ff_out = layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
                        x = residual + ff_out
                    else:
                        # Self-attention (Post-LN)
                        self_attn_out = self._multihead_attention(
                            x, k_full, v_full, layer.self_attn, need_weights=False
                        )[0]
                        x = layer.norm1(residual + self_attn_out)

                        # Cross-attention (Post-LN)
                        residual = x
                        cross_attn_out = layer.multihead_attn(
                            x, memory, memory,
                            key_padding_mask=memory_key_padding_mask, need_weights=False
                        )[0]
                        x = layer.norm2(residual + cross_attn_out)

                        # Feed-forward (Post-LN)
                        residual = x
                        ff_out = layer.linear2(layer.activation(layer.linear1(x)))
                        x = layer.norm3(residual + ff_out)

                x = self.decoder.norm(x)
            else:
                tgt_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
                )
                x = self.decoder(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
            
            x = x[:, -1:]  # 마지막 토큰의 출력만 사용
        
            logits = self.head(x.squeeze(1))
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_token = torch.where(finished.unsqueeze(1), eos_id, next_token)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all(): break

        return generated

    def _compute_kv(self, x, attention_module):
        """입력 x로부터 K, V 계산"""
        # MultiheadAttention의 in_proj를 사용하여 K, V 계산
        batch_size, seq_len, d_model = x.shape
        head_dim = self.hidden_size // self.num_heads
        
        # in_proj_weight: [3*d_model, d_model] -> [Q, K, V]
        w_q, w_k, w_v = attention_module.in_proj_weight.chunk(3)
        
        k = torch.nn.functional.linear(x, w_k, attention_module.in_proj_bias.chunk(3)[1] if attention_module.in_proj_bias is not None else None)
        v = torch.nn.functional.linear(x, w_v, attention_module.in_proj_bias.chunk(3)[2] if attention_module.in_proj_bias is not None else None)
        
        # Reshape to [batch, seq, num_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim)
        
        return k, v
    
    def _multihead_attention(self, q, k, v, attention_module, need_weights=False):
        """수동으로 multihead attention 계산 (K, V는 이미 계산된 형태)"""
        batch_size, q_len, d_model = q.shape
        _, kv_len, _, head_dim = k.shape
        
        # Q 계산
        w_q = attention_module.in_proj_weight.chunk(3)[0]
        b_q = attention_module.in_proj_bias.chunk(3)[0] if attention_module.in_proj_bias is not None else None
        q = torch.nn.functional.linear(q, w_q, b_q)
        q = q.view(batch_size, q_len, self.num_heads, head_dim).transpose(1, 2)
        
        # K, V reshape
        k = k.transpose(1, 2)  # [batch, heads, kv_len, head_dim]
        v = v.transpose(1, 2)  # [batch, heads, kv_len, head_dim]
        
        # Attention
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            attn_weights = None
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, d_model)
        
        # Output projection
        attn_output = torch.nn.functional.linear(
            attn_output, attention_module.out_proj.weight, attention_module.out_proj.bias
        )
        
        return attn_output, attn_weights if need_weights else None

if __name__ == "__main__":
    torch.manual_seed(0)

    B = 2
    T = 8
    A = 100
    S = 12
    K = 50
    D = 1024
    Depth = 4
    n_head = 4

    model = BasicTransformer(
        vocab_size=K,
        hidden_size=D,
        depth=Depth,
        num_heads=n_head,
        audio_dim=D,
        text_dim=D,
        max_output_length=32,
        dropout=0.1,
    )
    # trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    input_ids = torch.randint(0, K, (B, T))
    audio_feats = torch.randn(B, A, D)
    text_feats = torch.randn(B, S, D)
    audio_mask = torch.zeros(B, A, dtype=torch.bool)
    text_mask = torch.zeros(B, S, dtype=torch.bool)

    logits = model(
        input_ids=input_ids,
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
    )
    print(f"logits: {logits.shape}")

    generated = model.decode(
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
        max_output_length=16,
        sos_id=1,
        eos_id=2,
        use_cache=False,
    )
    print(f"generated: {generated.shape}")

    MAX_OUTPUT_LENGTH = 1024
    # Compare use_cache=True vs use_cache=False
    print("\n" + "="*50)
    print("Comparing use_cache=True vs use_cache=False")
    print("="*50)
    
    import time
    
    # Test with use_cache=False
    start_time = time.time()
    generated_no_cache = model.decode(
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
        max_output_length=MAX_OUTPUT_LENGTH,
        sos_id=1,
        eos_id=2,
        use_cache=False,
    )
    time_no_cache = time.time() - start_time
    print(f"use_cache=False - Time: {time_no_cache:.4f}s, Shape: {generated_no_cache.shape}")
    
    # Test with use_cache=True
    start_time = time.time()
    generated_with_cache = model.decode(
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
        max_output_length=MAX_OUTPUT_LENGTH,
        sos_id=1,
        eos_id=2,
        use_cache=True,
    )
    time_with_cache = time.time() - start_time
    print(f"use_cache=True  - Time: {time_with_cache:.4f}s, Shape: {generated_with_cache.shape}")
    
    # Calculate speedup
    speedup = time_no_cache / time_with_cache
    print(f"\nSpeedup (use_cache=True / use_cache=False): {speedup:.2f}x")
    print(f"Time saved: {(time_no_cache - time_with_cache):.4f}s ({(1 - time_with_cache/time_no_cache)*100:.1f}%)")
    
    # Check if outputs are identical
    output_match = (generated_no_cache == generated_with_cache).all()
    print(f"Outputs match: {output_match}")
    print("="*50 + "\n")