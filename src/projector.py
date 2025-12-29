import torch
import torch.nn as nn

class CTCProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            # layer 2            
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            # output layer
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:               
        return self.net(x)