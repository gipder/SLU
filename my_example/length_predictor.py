import torch
import torch.nn as nn
import torch.nn.functional as F

class LengthPredictor(nn.Module):
    """
    Length predictor for NAR:
      inputs:
        emb_seq:  [B, T, D]
        emb_mask: [B, T]  (1/True = valid, 0/False = pad)
      outputs:
        logits:   [B, Lmax+1]  (length class 0..Lmax)
    """
    def __init__(
        self,
        d_model: int,
        Lmax: int,
        hidden: int = 512,
        dropout: float = 0.1,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        self.Lmax = Lmax
        self.use_attn_pool = use_attn_pool

        # pooling
        if use_attn_pool:
            # attention pooling: score_t = w^T tanh(Wx_t)
            self.pool_proj = nn.Linear(d_model, d_model)
            self.pool_score = nn.Linear(d_model, 1)

        # classifier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, Lmax + 1)

    def _masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D], mask: [B,T] (bool or 0/1)
        mask_f = mask.to(dtype=x.dtype)  # [B,T]
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        pooled = (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]
        return pooled

    def _masked_attn_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        # scores: [B,T,1]
        h = torch.tanh(self.pool_proj(x))
        scores = self.pool_score(h).squeeze(-1)  # [B,T]

        # mask padded positions to -inf
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [B,T]
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B,D]
        return pooled

    def forward(self, emb_seq: torch.Tensor, emb_mask: torch.Tensor) -> torch.Tensor:
        """
        returns logits: [B, Lmax+1]
        """
        if self.use_attn_pool:
            pooled = self._masked_attn_pool(emb_seq, emb_mask)
        else:
            pooled = self._masked_mean_pool(emb_seq, emb_mask)

        h = self.mlp(pooled)
        logits = self.out(h)
        return logits

    @torch.no_grad()
    def predict(self, emb_seq: torch.Tensor, emb_mask: torch.Tensor) -> torch.Tensor:
        """
        returns predicted length (argmax): [B]
        """
        logits = self.forward(emb_seq, emb_mask)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def topk(self, emb_seq: torch.Tensor, emb_mask: torch.Tensor, k: int = 5):
        """
        returns (topk_lengths [B,k], topk_probs [B,k])
        """
        logits = self.forward(emb_seq, emb_mask)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(k, dim=-1)
        return topk_idx, topk_probs
