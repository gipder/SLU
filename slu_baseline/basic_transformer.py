import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class TimestepEmbedder(nn.Module):
    """
    Sinusoidal Positional Embedding for Time step t.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t, max_period=10000):
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


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
        use_timestep_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_output_length, hidden_size))

        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.modality_emb = nn.Parameter(torch.zeros(2, hidden_size))
        
        # Time step embedding for diffusion models
        self.use_timestep_embedding = use_timestep_embedding             
        if use_timestep_embedding:
            self.t_embedder = TimestepEmbedder(hidden_size)

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
        t: Optional[torch.Tensor] = None,
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
        
        # Add time step embedding if provided
        if t is not None:
            if not self.use_timestep_embedding:
                raise ValueError("use_timestep_embedding must be True in __init__ to use t")
            t_feat = self.t_embedder(t)  # (B, hidden_size)
            x = x + t_feat.unsqueeze(1)  # Broadcast to all positions
        
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
        use_timestep_embedding=True,
    )
    # trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    input_ids = torch.randint(0, K, (B, T))
    audio_feats = torch.randn(B, A, D)
    text_feats = torch.randn(B, S, D)
    audio_mask = torch.zeros(B, A, dtype=torch.bool)
    text_mask = torch.zeros(B, S, dtype=torch.bool)
    t = torch.tensor([0.1, 0.9])  # timestep in [0, 1]

    # Test forward with time embedding
    logits = model(
        input_ids=input_ids,
        t=t,
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,        
    )
    print(f"logits: {logits.shape}")    