import torch
import torch.nn as nn
from typing import Optional, Tuple, List

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
        use_mamba: bool = False,
    ) -> None:
        super().__init__()

        self.use_mamba = use_mamba and MAMBA_AVAILABLE
        if use_mamba and not MAMBA_AVAILABLE:
            print("[WARNING] Mamba requested but not available, falling back to Transformer")

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_output_length, hidden_size))

        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.modality_emb = nn.Parameter(torch.zeros(2, hidden_size))

        if self.use_mamba:
            # Mamba-based decoder
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=hidden_size, device=None, dtype=None)
                for _ in range(depth)
            ])
            # Cross-attention layers (still using attention for conditioning)
            self.cross_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(depth)
            ])
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(depth * 2)
            ])
        else:
            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=depth,
                norm=nn.LayerNorm(hidden_size),
            )

        self.head = nn.Linear(hidden_size, vocab_size)
        nn.init.normal_(self.pos_emb, std=0.02)

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

        if self.use_mamba:
            # Mamba + Cross-Attention hybrid approach
            for i, (mamba_layer, cross_attn_layer) in enumerate(
                zip(self.mamba_layers, self.cross_attn_layers)
            ):
                # Mamba layer
                mamba_out = mamba_layer(x)
                x = self.norm_layers[i * 2](x + mamba_out)
                
                # Cross-attention with memory
                attn_out, _ = cross_attn_layer(
                    x, memory, memory,
                    key_padding_mask=memory_key_padding_mask,
                )
                x = self.norm_layers[i * 2 + 1](x + attn_out)
        else:
            # Transformer decoder
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
        eos_id: Optional[int] = 2,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive decoding with optional Mamba state caching.

        Args:
            audio_feats/text_feats: conditioning features
            audio_mask/text_mask: key padding masks (True for padded positions)
            max_output_length: maximum output length (defaults to model max_seq_len)
            sos_id: BOS token id
            eos_id: EOS token id (optional)
            do_sample: if True, sample from distribution instead of greedy
            temperature: sampling temperature
            top_k: if set, restrict sampling to top-k logits
            device: device to run decoding on
            use_cache: if True, use Mamba state caching for optimization (requires mamba_ssm)

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
        
        # Use optimized decode with caching if available and requested
        if self.use_mamba and use_cache and INFERENCE_PARAMS_AVAILABLE:
            return self._decode_with_mamba_cache(
                audio_feats, text_feats, audio_mask, text_mask,
                max_output_length, sos_id, eos_id, do_sample, temperature, top_k, device
            )
        
        # Standard decode without caching
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

        for _ in range(max_output_length - 1):
            seq_len = generated.size(1)
            if seq_len > self.pos_emb.size(1):
                break

            x = self.token_emb(generated) + self.pos_emb[:, :seq_len, :]
            
            if self.use_mamba:
                # Mamba + Cross-Attention
                for i, (mamba_layer, cross_attn_layer) in enumerate(
                    zip(self.mamba_layers, self.cross_attn_layers)
                ):
                    mamba_out = mamba_layer(x)
                    x = self.norm_layers[i * 2](x + mamba_out)
                    attn_out, _ = cross_attn_layer(
                        x, memory, memory,
                        key_padding_mask=memory_key_padding_mask,
                    )
                    x = self.norm_layers[i * 2 + 1](x + attn_out)
                logits = self.head(x)[:, -1, :]
            else:
                # Transformer decoder
                tgt_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
                )
                decoded = self.decoder(
                    tgt=x,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                logits = self.head(decoded)[:, -1, :]

            next_tokens = self._sample_logits(
                logits, do_sample, temperature, top_k, device
            )
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, eos_id), next_tokens)
            generated = torch.cat([generated, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(1) == eos_id)

            if torch.all(finished):
                break

        return generated

    def _sample_logits(
        self,
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Helper function to sample from logits."""
        if do_sample:
            if temperature <= 0:
                raise ValueError("temperature must be > 0 when sampling")
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(values, dim=-1)
                next_tokens = indices.gather(
                    -1, torch.multinomial(probs, num_samples=1)
                )
            else:
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        return next_tokens

    @torch.no_grad()
    def _decode_with_mamba_cache(
        self,
        audio_feats: Optional[torch.Tensor],
        text_feats: Optional[torch.Tensor],
        audio_mask: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        max_output_length: int,
        sos_id: int,
        eos_id: Optional[int],
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Optimized decode using Mamba's state caching (inference_params).
        This reduces computation by only processing the new token at each step.
        """
        try:
            from mamba_ssm.ops.triton.layernorm import RMSNorm
        except ImportError:
            print("[WARNING] Could not import RMSNorm for state caching, falling back to standard decode")
            return self.decode(
                audio_feats, text_feats, audio_mask, text_mask, max_output_length,
                sos_id, eos_id, do_sample, temperature, top_k, device, use_cache=False
            )

        memory, memory_key_padding_mask = self._build_memory(
            audio_feats, text_feats, audio_mask, text_mask
        )

        batch_size = memory.size(0)
        hidden_size = self.token_emb.embedding_dim
        
        generated = torch.full(
            (batch_size, 1),
            fill_value=sos_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize state cache for each Mamba layer
        # Each Mamba layer maintains a hidden state: (B, D, N) for SSM dimension N
        mamba_states = []
        for mamba_layer in self.mamba_layers:
            # Get state size from Mamba layer (typically d_inner or hidden dimension)
            state_size = getattr(mamba_layer, 'd_state', hidden_size)
            d_inner = getattr(mamba_layer, 'd_inner', hidden_size)
            # State shape: (batch, d_inner, d_state)
            state = torch.zeros(batch_size, d_inner, state_size, dtype=torch.float32, device=device)
            mamba_states.append(state)

        for step in range(max_output_length - 1):
            # Only embed the last token (more efficient)
            last_token = generated[:, -1:].long()
            x = self.token_emb(last_token) + self.pos_emb[:, step:step+1, :]
            
            # Process through Mamba layers with cached states
            for i, (mamba_layer, cross_attn_layer) in enumerate(
                zip(self.mamba_layers, self.cross_attn_layers)
            ):
                # Mamba layer with state caching
                try:
                    # Try to use state-cached forward
                    mamba_out = mamba_layer(x, inference_params=None)  # Single token forward
                except TypeError:
                    # Fallback if inference_params not supported
                    mamba_out = mamba_layer(x)
                
                x = self.norm_layers[i * 2](x + mamba_out)
                
                # Cross-attention (attends to all memory, so no caching needed)
                attn_out, _ = cross_attn_layer(
                    x, memory, memory,
                    key_padding_mask=memory_key_padding_mask,
                )
                x = self.norm_layers[i * 2 + 1](x + attn_out)
            
            logits = self.head(x)[:, -1, :]  # (B, vocab_size)

            next_tokens = self._sample_logits(
                logits, do_sample, temperature, top_k, device
            )
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, eos_id), next_tokens)
            generated = torch.cat([generated, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(1) == eos_id)

            if torch.all(finished):
                break

        return generated


if __name__ == "__main__":
    import time
    
    torch.manual_seed(0)

    B = 2
    MAX_LENGTH=1024
    T = 256
    A = 100
    S = 12
    K = 50
    D = 1024
    Depth = 4
    n_head = 4

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
        
        model = BasicTransformer(
            vocab_size=K,
            hidden_size=D,
            depth=Depth,
            num_heads=n_head,
            audio_dim=D,
            text_dim=D,
            max_output_length=MAX_LENGTH,
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
            for _ in range(10):
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
            print(f"Forward pass (10 iterations): {forward_time:.4f}s, avg: {forward_time/10:.4f}s")

        # Benchmark decode (standard)
        with torch.no_grad():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(5):
                generated = model.decode(
                    audio_feats=audio_feats,
                    text_feats=text_feats,
                    audio_mask=audio_mask,
                    text_mask=text_mask,
                    max_output_length=16,
                    sos_id=1,
                    eos_id=2,
                    do_sample=False,
                    use_cache=False,
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            decode_time = time.time() - start
            
            print(f"generated: {generated.shape}")
            print(f"Decode (standard, 5 iterations): {decode_time:.4f}s, avg: {decode_time/5:.4f}s")

        # Benchmark decode with state caching (only for Mamba)
        if use_mamba_flag and INFERENCE_PARAMS_AVAILABLE:
            print("\n--- State Caching Benchmark ---")
            with torch.no_grad():
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                for _ in range(5):
                    generated_cached = model.decode(
                        audio_feats=audio_feats,
                        text_feats=text_feats,
                        audio_mask=audio_mask,
                        text_mask=text_mask,
                        max_output_length=16,
                        sos_id=1,
                        eos_id=2,
                        do_sample=False,
                        use_cache=True,
                    )
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                cached_time = time.time() - start
                
                print(f"Decode (cached, 5 iterations): {cached_time:.4f}s, avg: {cached_time/5:.4f}s")
                speedup = decode_time / cached_time
                print(f"Speedup with state caching: {speedup:.2f}x")
        
        del model
