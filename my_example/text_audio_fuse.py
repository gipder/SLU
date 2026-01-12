import torch
import torch.nn as nn
from typing import Optional, Tuple

class TextAudioFusePool(nn.Module):
    """
    emb_attn  = MHA(emb_text, emb_aud, emb_aud)   # cross-attn
    emb_stack = Stack(emb_text, emb_attn)         # concat on feature dim
    emb_fused = Linear(emb_stack)
    emb_out   = Transformer(emb_fused)            # self-attn encoder
    emb_pool  = masked_mean_pool(emb_out, text_mask)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        d_out: int = 512,                        
        n_layers: int = 2,
        dropout: float = 0.1,
        pool: str = "mean",  # "mean" or "cls"
        is_proj: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert pool in ("mean", "cls")

        self.d_model = d_model
        self.pool = pool

        # cross-attention: Q=text, K/V=audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )

        # Stack(concat) then Linear
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

        # Transformer over fused text-length sequence
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # optional CLS pooling
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # output projection
        self.is_proj = is_proj
        self.proj = nn.Linear(d_model, d_out)

    @staticmethod
    def _to_key_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        """
        valid_mask: Bool/0-1 mask with True(1)=valid, False(0)=pad, shape [B, T]
        return: key_padding_mask where True means PAD (as PyTorch expects)
        """
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask != 0
        return ~valid_mask  # True at PAD positions

    @staticmethod
    def masked_mean_pool(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        x: [B, T, D]
        valid_mask: [B, T] (True=valid)
        return: [B, D]
        """
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask != 0
        w = valid_mask.unsqueeze(-1).type_as(x)  # [B, T, 1]
        x_sum = (x * w).sum(dim=1)
        denom = w.sum(dim=1).clamp_min(eps)
        return x_sum / denom

    def forward(
        self,
        emb_text: torch.Tensor,          # [B, Lt, D]
        text_mask: torch.Tensor,         # [B, Lt] True/1=valid
        emb_audio: torch.Tensor,         # [B, La, D]
        audio_mask: torch.Tensor,        # [B, La] True/1=valid
        need_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          emb_pool: [B, D] pooled vector
          emb_seq:  [B, Lt, D] (or [B, Lt+1, D] if pool="cls")
          attn_w:   [B, Lt, La] attention weights (optional, averaged over heads)
        """
        B, Lt, D = emb_text.shape
        assert D == self.d_model and emb_audio.size(-1) == self.d_model

        # PyTorch MHA masks: key_padding_mask uses True for PAD
        audio_kpm = self._to_key_padding_mask(audio_mask)  # [B, La]

        # (Optional) block PAD queries too: attn_mask per-batch isn’t supported nicely for query pads,
        # so we’ll just zero them later via text_mask.
        emb_attn, attn_w = self.cross_attn(
            query=emb_text,
            key=emb_audio,
            value=emb_audio,
            key_padding_mask=audio_kpm,
            need_weights=need_attn_weights,
            average_attn_weights=True,  # attn_w: [B, Lt, La] if need_weights
        )

        # Stack/concat on feature dim
        emb_stack = torch.cat([emb_text, emb_attn], dim=-1)  # [B, Lt, 2D]
        emb_fused = self.fuse(emb_stack)                     # [B, Lt, D]

        # If using CLS pooling, prepend CLS and update mask
        if self.pool == "cls":
            cls = self.cls_token.expand(B, 1, D)             # [B, 1, D]
            emb_fused = torch.cat([cls, emb_fused], dim=1)   # [B, 1+Lt, D]
            if text_mask.dtype != torch.bool:
                text_mask = text_mask != 0
            cls_mask = torch.ones(B, 1, device=text_mask.device, dtype=torch.bool)
            enc_mask_valid = torch.cat([cls_mask, text_mask], dim=1)  # [B, 1+Lt]
        else:
            enc_mask_valid = text_mask

        enc_kpm = self._to_key_padding_mask(enc_mask_valid)  # [B, Tenc]

        # Transformer encoder (self-attn)
        emb_seq = self.encoder(emb_fused, src_key_padding_mask=enc_kpm)  # [B, Tenc, D]

        # Pool
        if self.pool == "cls":
            emb_pool = emb_seq[:, 0]  # [B, D]
        else:
            emb_pool = self.masked_mean_pool(emb_seq, enc_mask_valid)  # [B, D]

        # Ensure PAD query positions don’t leak (optional cleanup)
        if self.pool == "mean":
            if text_mask.dtype != torch.bool:
                text_mask = text_mask != 0
            emb_seq = emb_seq * text_mask.unsqueeze(-1).type_as(emb_seq)
        
        if self.is_proj:
            emb_pool = self.proj(emb_pool)
            emb_seq = self.proj(emb_seq)

        return emb_pool, emb_seq, attn_w if need_attn_weights else None


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    B, Lt, La, D = 2, 5, 7, 256
    emb_text = torch.randn(B, Lt, D).to('cuda')
    emb_audio = torch.randn(B, La, D).to('cuda')

    # True/1=valid, False/0=pad
    text_mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,1]]).bool().to('cuda')
    audio_mask = torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,0]]).bool().to('cuda')

    model = TextAudioFusePool(d_model=D, n_heads=8, n_layers=2, pool="mean")
    model = model.to('cuda')
    emb_pool, emb_seq, attn_w = model(emb_text, text_mask, emb_audio, audio_mask, need_attn_weights=True)

    print(emb_pool.shape)  # [B, D]
    print(emb_seq.shape)   # [B, Lt, D]
    print(attn_w.shape)
