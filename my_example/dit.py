import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def _make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # [T, T] with True = masked positions
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def _expand_kv_mask(kv_mask: Optional[torch.Tensor], T_q: int) -> Optional[torch.Tensor]:
    """
    kv_mask: [B, T_k] where 1/True = keep, 0/False = pad
    return attn_mask shaped for SDPA: [B, 1, T_q, T_k] bool, True=mask
    """
    if kv_mask is None:
        return None
    if kv_mask.dtype != torch.bool:
        kv_mask = kv_mask.bool()
    # True means "mask out" in SDPA
    return (~kv_mask)[:, None, None, :].expand(-1, 1, T_q, -1)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for continuous t (0..1 or any float)."""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] float
        returns: [B, dim]
        """
        half = self.dim // 2
        device = t.device
        # log-spaced frequencies
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / (half - 1)
        )  # [half]
        args = t[:, None] * freqs[None, :]  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D], shift/scale: [B, D]
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


# -----------------------------
# Attention blocks (SDPA)
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, D]
        attn_mask: optional bool mask for SDPA:
          - either [T, T] (causal) or [B, 1, T, T] (padding)
          - True means "masked out"
        """
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, h, T, hd]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # SDPA expects float masks? In torch, bool mask is OK: True=mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, h, T, hd]

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.proj(y)
        return y


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim_q % n_heads == 0
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.n_heads = n_heads
        self.head_dim = dim_q // n_heads

        self.q = nn.Linear(dim_q, dim_q, bias=True)
        self.k = nn.Linear(dim_kv, dim_q, bias=True)
        self.v = nn.Linear(dim_kv, dim_q, bias=True)
        self.proj = nn.Linear(dim_q, dim_q, bias=True)
        self.dropout = dropout

    def forward(
        self,
        x_q: torch.Tensor,          # [B, Tq, Dq]
        x_kv: torch.Tensor,         # [B, Tk, Dkv]
        kv_attn_mask: Optional[torch.Tensor] = None,  # [B, 1, Tq, Tk] bool
    ) -> torch.Tensor:
        B, Tq, Dq = x_q.shape
        Tk = x_kv.size(1)

        q = self.q(x_q)
        k = self.k(x_kv)
        v = self.v(x_kv)

        q = q.view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)  # [B,h,Tq,hd]
        k = k.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)  # [B,h,Tk,hd]
        v = v.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=kv_attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, Tq, Dq)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


# -----------------------------
# DiT-style Transformer block (adaLN-Zero)
# -----------------------------
class DiTSeqBlock(nn.Module):
    """
    DiT-like block with:
      - self-attn on output tokens
      - cross-attn to condition tokens (emb_seq)
      - MLP
      - adaLN-Zero conditioning using time embedding
    """
    def __init__(self, dim: int, cond_dim: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)

        self.self_attn = MultiHeadSelfAttention(dim, n_heads, dropout=dropout)
        self.cross_attn = MultiHeadCrossAttention(dim, cond_dim, n_heads, dropout=dropout)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        # adaLN-Zero parameters from time embedding:
        # For each sublayer we produce shift, scale, gate (3*D).
        self.ada1 = nn.Linear(dim, 3 * dim)
        self.ada2 = nn.Linear(dim, 3 * dim)
        self.ada3 = nn.Linear(dim, 3 * dim)

        # init "zero" so the network starts near identity (DiT trick)
        nn.init.zeros_(self.ada1.weight); nn.init.zeros_(self.ada1.bias)
        nn.init.zeros_(self.ada2.weight); nn.init.zeros_(self.ada2.bias)
        nn.init.zeros_(self.ada3.weight); nn.init.zeros_(self.ada3.bias)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T_out, D]
        t_emb: torch.Tensor,             # [B, D]
        cond: torch.Tensor,              # [B, T_in, D_cond]
        cond_mask: Optional[torch.Tensor] = None,  # [B, T_in] keep=1
        self_causal: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # --- self-attn ---
        s1, sc1, g1 = self.ada1(t_emb).chunk(3, dim=-1)  # [B,D] x3
        h = modulate(self.norm1(x), s1, sc1)

        attn_mask = None
        if self_causal:
            attn_mask = _make_causal_mask(T, x.device)  # [T,T] bool

        h = self.self_attn(h, attn_mask=attn_mask)
        x = x + (g1[:, None, :]) * h

        # --- cross-attn ---
        s2, sc2, g2 = self.ada2(t_emb).chunk(3, dim=-1)
        h = modulate(self.norm2(x), s2, sc2)

        kv_mask = _expand_kv_mask(cond_mask, T_q=T)  # [B,1,T,Tin] or None
        h = self.cross_attn(h, cond, kv_attn_mask=kv_mask)
        x = x + (g2[:, None, :]) * h

        # --- MLP ---
        s3, sc3, g3 = self.ada3(t_emb).chunk(3, dim=-1)
        h = modulate(self.norm3(x), s3, sc3)
        h = self.mlp(h)
        x = x + (g3[:, None, :]) * h

        return x


# -----------------------------
# Full model
# -----------------------------
@dataclass
class DiTSeq2SeqConfig:
    K: int                 # vocab/classes for output logits
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_T_out: int = 512   # maximum length you will ever decode
    cond_in_dim: int = 512 # D_in of emb_seq
    self_causal: bool = False  # if you want autoregressive-style self-attn


class DiTSeq2Seq(nn.Module):
    def __init__(self, cfg: DiTSeq2SeqConfig):
        super().__init__()
        self.cfg = cfg

        self.cond_proj = nn.Linear(cfg.cond_in_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(cfg.max_T_out, cfg.d_model) * 0.02)

        self.time_sin = SinusoidalTimeEmbedding(cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.SiLU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )

        # ★ 추가: x_t 임베딩
        # (1) x_t가 [B,T] long 인 경우
        self.x_token_emb = nn.Embedding(cfg.K, cfg.d_model)

        # (2) x_t가 [B,T,K] one-hot/prob 인 경우를 지원하려면 아래도 추가
        # self.x_onehot_proj = nn.Linear(cfg.K, cfg.d_model, bias=False)

        self.blocks = nn.ModuleList([
            DiTSeqBlock(cfg.d_model, cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.K)

    def forward(
        self,
        x_t: torch.Tensor,                 # ★ [B,T_out] long  또는 [B,T_out,K] float
        t: torch.Tensor,                   # [B]
        emb_seq: torch.Tensor,             # [B,T_in,D_in] (condition)
        emb_mask: Optional[torch.Tensor],  # [B,T_in] keep=1        
    ) -> torch.Tensor:
        """
        returns logits: [B, T_out, K]
        """
        B = emb_seq.size(0)
        T_out = x_t.size(1)
        assert T_out <= self.cfg.max_T_out

        # condition
        cond = self.cond_proj(emb_seq)  # [B,T_in,d_model]
        if emb_mask is not None and emb_mask.dtype != torch.bool:
            emb_mask = emb_mask.bool()

        # time embedding
        if t.dim() != 1:
            t = t.view(-1)
        t_emb = self.time_mlp(self.time_sin(t))  # [B,d_model]

        # ★ x_t -> model space
        if x_t.dim() == 2:
            # x_t: [B,T] long
            x = self.x_token_emb(x_t)  # [B,T,d_model]
        elif x_t.dim() == 3:
            # x_t: [B,T,K] float (one-hot/prob)
            # 방법 A: 임베딩 테이블과 matmul (권장: one-hot/prob 모두 자연스럽게 처리)
            x = x_t @ self.x_token_emb.weight  # [B,T,d_model]
            # 방법 B: Linear proj를 따로 두고 싶으면:
            # x = self.x_onehot_proj(x_t)
        else:
            raise ValueError(f"x_t must be [B,T] or [B,T,K], got {tuple(x_t.shape)}")

        # add positional embedding
        x = x + self.pos_emb[:T_out].unsqueeze(0)

        # transformer blocks
        for blk in self.blocks:
            x = blk(
                x=x,
                t_emb=t_emb,
                cond=cond,
                cond_mask=emb_mask,
                self_causal=self.cfg.self_causal,
            )

        x = self.final_norm(x)
        logits = self.head(x)  # [B,T_out,K]
        return logits


# -----------------------------
# Minimal usage example
# -----------------------------
if __name__ == "__main__":
    B, T_in, D_in = 2, 21, 1024
    T_out = 30
    K = 650

    cfg = DiTSeq2SeqConfig(K=K, d_model=512, n_layers=6, n_heads=8, max_T_out=128, cond_in_dim=D_in)
    model = DiTSeq2Seq(cfg)
    
    emb_seq = torch.randn(B, T_in, D_in)
    emb_mask = torch.ones(B, T_in, dtype=torch.bool)
    x0 = torch.randint(0, K, (B, T_out))
    t = torch.rand(B)  # e.g., diffusion time in (0,1)

    logits = model(x0, t, emb_seq, emb_mask)
    print(logits.shape)  # (B, T_out, K)
