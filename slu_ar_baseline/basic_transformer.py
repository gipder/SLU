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
    ) -> None:
        super().__init__()

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
            norm_first=True,  # Use Pre-LN for better stability with deep models
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
        eos_id: Optional[int] = 2,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
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

        for _ in range(max_output_length - 1):
            seq_len = generated.size(1)
            if seq_len > self.pos_emb.size(1):
                break

            x = self.token_emb(generated) + self.pos_emb[:, :seq_len, :]
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

            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, eos_id), next_tokens)
            generated = torch.cat([generated, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(1) == eos_id)

            if torch.all(finished):
                break
            """
            if eos_id is not None:
                if torch.all(next_tokens.squeeze(1) == eos_id):
                    break
            """

        return generated


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
        do_sample=False,
    )
    print(f"generated: {generated.shape}")
