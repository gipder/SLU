import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import time

# FLOPs 계산을 위한 라이브러리
try:
    from thop import profile
    FLOPS_COUNTER_AVAILABLE = True
except ImportError:
    try:
        from fvcore.nn import FlopCounter
        FLOPS_COUNTER_AVAILABLE = True
    except ImportError:
        FLOPS_COUNTER_AVAILABLE = False
        print("[INFO] Neither thop nor fvcore available, FLOPs calculation will be skipped")

# --- Mamba Compatibility Layer ---
try:
    from mamba_ssm import Mamba
    # 최신 버전은 layer_norm (언더바 있음)을 사용합니다.
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except ImportError:
        from mamba_ssm.ops.triton.layernorm import RMSNorm
    
    # 추론 최적화를 위한 핵심 객체
    from mamba_ssm.utils.generation import InferenceParams
    
    MAMBA_AVAILABLE = True
    INFERENCE_PARAMS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    MAMBA_AVAILABLE = False
    INFERENCE_PARAMS_AVAILABLE = False
    print(f"[INFO] Mamba specialized ops not found, using fallback. Error: {e}")

    # Fallback RMSNorm implementation
    class RMSNorm(nn.Module):
        def __init__(self, d_model, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))
        def forward(self, x):
            output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return output * self.weight

# ----------------------------------

class SimpleMamba(nn.Module):
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
        use_mamba: bool = True,
    ) -> None:
        super().__init__()

        self.use_mamba = use_mamba and MAMBA_AVAILABLE
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_output_length, hidden_size))

        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.modality_emb = nn.Parameter(torch.zeros(2, hidden_size))

        if self.use_mamba:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2, layer_idx=i)
                for i in range(depth)
            ])
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(depth)
            ])
            self.cond_proj = nn.Linear(hidden_size, hidden_size)
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
                dropout=dropout, batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth, norm=nn.LayerNorm(hidden_size))
            
            # Transformer용 self-attention KV 캐시 구성 (각 layer별 head별)
            self.self_attn_kv_cache = None

        self.head = nn.Linear(hidden_size, vocab_size)
        nn.init.normal_(self.pos_emb, std=0.02)

    def _build_memory(self, audio_feats, text_feats, audio_mask, text_mask):
        memory_list, mask_list = [], []
        if audio_feats is not None:
            memory_list.append(self.audio_proj(audio_feats) + self.modality_emb[0])
            if audio_mask is not None: mask_list.append(audio_mask)
        if text_feats is not None:
            memory_list.append(self.text_proj(text_feats) + self.modality_emb[1])
            if text_mask is not None: mask_list.append(text_mask)
        
        memory = torch.cat(memory_list, dim=1)
        mask = torch.cat(mask_list, dim=1) if mask_list else None
        return memory, mask

    def _pool_memory(self, memory, memory_mask):
        if memory is None:
            return None
        if memory_mask is None:
            pooled = memory.mean(dim=1)
        else:
            # memory_mask: True for padding
            valid = (~memory_mask).unsqueeze(-1).type_as(memory)
            denom = valid.sum(dim=1).clamp_min(1.0)
            pooled = (memory * valid).sum(dim=1) / denom
        return self.cond_proj(pooled)

    def forward(self, input_ids, audio_feats=None, text_feats=None, audio_mask=None, text_mask=None):
        seq_len = input_ids.shape[1]
        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        memory, memory_mask = self._build_memory(audio_feats, text_feats, audio_mask, text_mask)
        cond = self._pool_memory(memory, memory_mask) if self.use_mamba else None

        if self.use_mamba:
            if cond is not None:
                x = x + cond.unsqueeze(1)
            for i, mamba_layer in enumerate(self.mamba_layers):
                x = self.norm_layers[i](x + mamba_layer(x))
        else:
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 1)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        
        return self.head(x)

    @torch.no_grad()
    def decode(self, audio_feats=None, text_feats=None, audio_mask=None, text_mask=None, 
               max_output_length=16, sos_id=1, eos_id=2, use_cache=True, use_amp=False):
        device = audio_feats.device if audio_feats is not None else text_feats.device
        batch_size = (audio_feats if audio_feats is not None else text_feats).size(0)
        
        memory, memory_mask = self._build_memory(audio_feats, text_feats, audio_mask, text_mask)
        cond = self._pool_memory(memory, memory_mask) if self.use_mamba else None
        generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 캐시 초기화 (Mamba와 Transformer 공용)
        inference_params = None
        kv_cache = None
        
        if use_cache:
            if self.use_mamba:
                if INFERENCE_PARAMS_AVAILABLE:
                    # Mamba 상태 캐시
                    inference_params = InferenceParams(max_seqlen=max_output_length, max_batch_size=batch_size)
                else:
                    print("[WARN] Mamba inference cache unavailable; falling back to non-cached decoding.")
            else:
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

        autocast_ctx = torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda" and not self.use_mamba,
        )
        for step in range(max_output_length - 1):
            with autocast_ctx:
                if self.use_mamba and use_cache and inference_params:
                    # Mamba with state caching
                    curr_input = generated[:, -1:]
                    x = self.token_emb(curr_input) + self.pos_emb[:, step:step+1, :]

                    if cond is not None:
                        x = x + cond.unsqueeze(1)

                    for i, mamba in enumerate(self.mamba_layers):
                        x = self.norm_layers[i](x + mamba(x, inference_params=inference_params))

                    # Mamba 캐시 시퀀스 위치 업데이트
                    inference_params.seqlen_offset += 1


                elif not self.use_mamba and use_cache and kv_cache is not None:
                    # Transformer with KV caching
                    seq_len = generated.shape[1]
                    x = self.token_emb(generated[:, -1:]) + self.pos_emb[:, seq_len-1:seq_len, :]

                    for layer_idx, layer in enumerate(self.decoder.layers):
                        # Self-attention with KV cache
                        q = x
                        # 이전 K, V를 불러옴
                        prev_k = kv_cache['k'][layer_idx]  # [batch, prev_seq, heads, head_dim]
                        prev_v = kv_cache['v'][layer_idx]

                        # 현재 토큰의 K, V 계산
                        curr_k, curr_v = self._compute_kv(x, layer.self_attn)

                        # K, V 캐시 업데이트 (이전 + 현재)
                        k_full = torch.cat([prev_k, curr_k], dim=1) if prev_k.shape[1] > 0 else curr_k
                        v_full = torch.cat([prev_v, curr_v], dim=1) if prev_v.shape[1] > 0 else curr_v

                        kv_cache['k'][layer_idx] = k_full
                        kv_cache['v'][layer_idx] = v_full

                        # Self-attention (Q=현재, K,V=캐시된 전체)
                        self_attn_out = self._multihead_attention(
                            q, k_full, v_full, layer.self_attn, need_weights=False
                        )[0]
                        x = layer.norm1(x + layer.dropout1(self_attn_out))

                        # Cross-attention
                        cross_attn_out = layer.multihead_attn(
                            x, memory, memory, key_padding_mask=memory_mask, need_weights=False
                        )[0]
                        x = layer.norm2(x + layer.dropout2(cross_attn_out))

                        # Feed-forward
                        ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                        x = layer.norm3(x + layer.dropout3(ff_out))

                    x = self.decoder.norm(x)

                else:
                    # 캐시 미사용: 전체 시퀀스 재계산
                    seq_len = generated.shape[1]
                    x_full = self.token_emb(generated) + self.pos_emb[:, :seq_len, :]
                    if self.use_mamba:
                        x = x_full
                        if cond is not None:
                            x = x + cond.unsqueeze(1)
                        for i, mamba in enumerate(self.mamba_layers):
                            x = self.norm_layers[i](x + mamba(x))
                    else:
                        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)
                        x = self.decoder(x_full, memory, tgt_mask=mask, memory_key_padding_mask=memory_mask)
                    x = x[:, -1:]

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

# --- 실행 및 벤치마크 코드 ---
if __name__ == "__main__":
    import time
    
    torch.manual_seed(0)

    B = 2
    MAX_T = 256
    T = MAX_T
    A = 1024
    S = MAX_T
    K = 50
    D = 1024
    Depth = 4
    n_head = 4
    num_iters = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare test data
    input_ids = torch.randint(0, K, (B, T))
    audio_feats = torch.randn(B, A, D)
    text_feats = torch.randn(B, S, D)
    audio_mask = torch.zeros(B, A, dtype=torch.bool)
    text_mask = torch.zeros(B, S, dtype=torch.bool)

    # Move to device
    input_ids = input_ids.to(device)
    audio_feats = audio_feats.to(device)
    text_feats = text_feats.to(device)
    audio_mask = audio_mask.to(device)
    text_mask = text_mask.to(device)

    for use_mamba_flag in [False, True]:
        print(f"\n{'='*50}")
        print(f"Testing with use_mamba={use_mamba_flag}")
        print(f"{'='*50}")
        
        model = SimpleMamba(
            vocab_size=K,
            hidden_size=D,
            depth=Depth,
            num_heads=n_head,
            audio_dim=D,
            text_dim=D,
            max_output_length=MAX_T,
            dropout=0.1,
            use_mamba=use_mamba_flag,
        )
        
        model = model.to(device)
        model.eval()

        # trainable params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Actual use_mamba: {model.use_mamba}")

        # Benchmark forward pass
        with torch.no_grad():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(num_iters):
                logits = model(
                    input_ids=input_ids,
                    audio_feats=audio_feats,
                    text_feats=text_feats,
                    audio_mask=audio_mask,
                    text_mask=text_mask,
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_time = time.time() - start
            
            print(f"logits: {logits.shape}")
            print(f"Forward pass ({num_iters} iterations): {forward_time:.4f}s, avg: {forward_time/num_iters:.4f}s")
            
            # FLOPs 계산
            if FLOPS_COUNTER_AVAILABLE:
                try:
                    from thop import profile
                    flops, _ = profile(model, inputs=(
                        input_ids, audio_feats, text_feats, audio_mask, text_mask
                    ), verbose=False)
                    print(f"Forward FLOPs: {flops / 1e9:.2f}G per batch")
                except Exception as e:
                    print(f"[INFO] FLOPs calculation failed: {e}")

        def _bench_decode(use_cache: bool, warmup: int = 3, iters: int = 10) -> float:
            # Warmup
            for _ in range(warmup):
                _ = model.decode(
                    audio_feats=audio_feats,
                    text_feats=text_feats,
                    audio_mask=audio_mask,
                    text_mask=text_mask,
                    max_output_length=MAX_T,
                    sos_id=1,
                    eos_id=2,
                    use_cache=use_cache,
                )

            if torch.cuda.is_available():
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
                for _ in range(iters):
                    _ = model.decode(
                        audio_feats=audio_feats,
                        text_feats=text_feats,
                        audio_mask=audio_mask,
                        text_mask=text_mask,
                        max_output_length=MAX_T,
                        sos_id=1,
                        eos_id=2,
                        use_cache=use_cache,
                    )
                ender.record()
                torch.cuda.synchronize()
                total_ms = starter.elapsed_time(ender)
                return total_ms / 1000.0
            else:
                start = time.time()
                for _ in range(iters):
                    _ = model.decode(
                        audio_feats=audio_feats,
                        text_feats=text_feats,
                        audio_mask=audio_mask,
                        text_mask=text_mask,
                        max_output_length=MAX_T,
                        sos_id=1,
                        eos_id=2,
                        use_cache=use_cache,
                    )
                return time.time() - start

        # Benchmark decode (no cache vs cache)
        print("\n--- Decode Benchmark (no cache vs cache) ---")
        with torch.no_grad():
            decode_time = _bench_decode(use_cache=False, warmup=3, iters=num_iters)
            cached_time = _bench_decode(use_cache=True, warmup=3, iters=num_iters)

        avg_decode = decode_time / num_iters
        avg_cached = cached_time / num_iters
        print(f"Decode (no cache, {num_iters} iterations): {decode_time:.4f}s, avg: {avg_decode:.4f}s")
        print(f"Decode (with cache, {num_iters} iterations): {cached_time:.4f}s, avg: {avg_cached:.4f}s")
        speedup = decode_time / cached_time if cached_time > 0 else 0
        print(f"Speedup with caching: {speedup:.2f}x")
        
        del model
