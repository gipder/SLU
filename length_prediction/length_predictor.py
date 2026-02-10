# This code is from pytext of Meta.
# All rights reserved in Meta.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def mean(rep: Tensor, padding_mask: Optional[Tensor]):
    # rep: B x T x C
    rep_sum = rep.sum(dim=1)  # B x C
    if padding_mask is not None:
        # padding_mask: B x T (True = pad)
        lengths = (~padding_mask).sum(dim=1).reshape(-1, 1)  # B x 1        
    else:
        bsz, max_token_len, _embed_dim = rep.size()
        lengths = torch.full(
            (bsz, 1), max_token_len, dtype=torch.long, device=rep.device
        )

    return rep_sum / lengths.clamp(min=1)


def pool(pooling_type: str, words: Tensor, encoder_padding_mask: Optional[Tensor]):
    # words: B x T x C
    if pooling_type == "mean":
        return mean(words, encoder_padding_mask)
    elif pooling_type == "max":
        return words.max(dim=1)[0]  # B x C
    elif pooling_type == "none":
        return words
    else:
        raise NotImplementedError(f"Unknown pooling_type: {pooling_type}")


class TimeConv1d(nn.Module):
    """
    래퍼: 입력/출력은 T x B x C 형태를 유지하면서 내부에서 Conv1d(B, C, T) 사용.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: T x B x C
        T, B, C = x.shape
        x = x.permute(1, 2, 0)  # B x C x T
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # T x B x C
        return x


class ConvLengthPredictionModule(nn.Module):
    """
    원래 PyText/LightweightConv 기반 모듈을
    순수 PyTorch Conv1d 기반으로 다시 구현.
    """

    def __init__(
        self,
        embed_dim: int,
        conv_dim: int = 128,
        max_target_positions: int = 128,
        length_dropout: float = 0.2,
        glu: bool = True,
        activation: str = "glu",  # "glu" or "relu"
        pooling_type: str = "mean",
        kernel_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3]

        self.length_dropout = length_dropout
        self.glu = glu
        self.pooling_type = pooling_type

        # conv layers (T x B x C in/out)
        self.conv_layers = nn.ModuleList(
            [TimeConv1d(conv_dim, k) for k in kernel_sizes]
        )

        # linear projections
        if glu:
            self.linear1 = nn.Linear(embed_dim, 2 * conv_dim)
        else:
            self.linear1 = nn.Linear(embed_dim, conv_dim)
        self.linear2 = nn.Linear(conv_dim, embed_dim)

        # activation
        if activation.lower() == "glu":
            self.activation = lambda x: F.glu(x, dim=2)  # T x B x (2C) → T x B x C
        elif activation.lower() == "relu":
            self.activation = lambda x: F.relu(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # final length predictor
        self.lengths_pred = nn.Linear(embed_dim, max_target_positions)

    def forward(self, x: Tensor, encoder_padding_mask: Optional[Tensor] = None):
        """
        x: T x B x C
        encoder_padding_mask: B x T (True = pad)
        """
        for conv in self.conv_layers:
            residual = x
            x = self.linear1(x)          # T x B x (C or 2C)
            x = self.activation(x)       # T x B x conv_dim

            if encoder_padding_mask is not None:
                # mask: B x T → T x B x 1                
                if encoder_padding_mask.dtype != torch.bool:
                    encoder_padding_mask = encoder_padding_mask.bool()
                                        
                x = x.masked_fill(
                    encoder_padding_mask.unsqueeze(2), 0
                )

            x = conv(x)                  # T x B x conv_dim
            x = self.linear2(x)          # T x B x embed_dim
            x = F.dropout(
                x, p=self.length_dropout, training=self.training
            )
            x = residual + x             # residual connection

        if encoder_padding_mask is not None:
            x = x.masked_fill(
                encoder_padding_mask.unsqueeze(2), 0
            )

        #x = x.transpose(0, 1)  # T x B x C → B x T x C
        x = F.relu(x)
        #print(f"{x.shape=}")
        lengths_enc = pool(self.pooling_type, x, encoder_padding_mask)
        #print(f"{lengths_enc.shape=}")
        predicted_lengths_logits = self.lengths_pred(lengths_enc)
        #print(f"{predicted_lengths_logits.shape=}")
        predicted_lengths = F.log_softmax(predicted_lengths_logits, dim=-1)
        return predicted_lengths, predicted_lengths_logits

    def create_eval_module(self):
        # 원래 PyText 인터페이스 맞춰서 그냥 self 반환
        return self


class MaskedLengthPredictionModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        length_hidden_dim: int = 128,
        max_target_positions: int = 128,
        length_dropout: float = 0.2,
    ):
        super().__init__()
        self.lengths_linear = nn.Linear(embed_dim, length_hidden_dim)
        self.lengths_pred = nn.Linear(length_hidden_dim, max_target_positions)
        self.length_dropout = length_dropout

    def forward(
        self, x: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: B x T x C
        encoder_padding_mask: B x T (True = pad)        
        """        
        if encoder_padding_mask is not None:
            x = x.masked_fill(
                encoder_padding_mask.unsqueeze(2), 0
            )
        avg_enc = mean(x, encoder_padding_mask)  # B x C        
        lengths_enc = self.lengths_linear(avg_enc)
        lengths_enc = F.relu(lengths_enc)
        lengths_enc = F.dropout(
            lengths_enc, p=self.length_dropout, training=self.training
        )
        logits = self.lengths_pred(lengths_enc)
        
        return logits

    def create_eval_module(self):
        return self
    

class MaskedLengthRegressionModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        length_hidden_dim: int = 128,        
        length_dropout: float = 0.2,
    ):
        super().__init__()
        self.lengths_linear = nn.Linear(embed_dim, length_hidden_dim)
        self.lengths_pred = nn.Linear(length_hidden_dim, 1)
        self.length_dropout = length_dropout

    def forward(
        self, x: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: B x T x C
        encoder_padding_mask: B x T (True = pad)        
        """        
        if encoder_padding_mask is not None:
            x = x.masked_fill(
                encoder_padding_mask.unsqueeze(2), 0
            )
        avg_enc = mean(x, encoder_padding_mask)  # B x C        
        lengths_enc = self.lengths_linear(avg_enc)
        lengths_enc = F.relu(lengths_enc)
        lengths_enc = F.dropout(
            lengths_enc, p=self.length_dropout, training=self.training
        )
        logits = self.lengths_pred(lengths_enc)
        regression = F.relu(logits).squeeze(1)  # B

        return regression

    def create_eval_module(self):
        return self



# main function
if __name__ == "__main__":
    T, B, C = 50, 4, 256  # seq_len, batch, embed_dim
    x = torch.randn(B, T, C)
    padding_mask = torch.zeros(B, T, dtype=torch.bool)  # 패딩 없다고 가정

    model = ConvLengthPredictionModule(
        embed_dim=C,
        conv_dim=256,
        max_target_positions=128,
        length_dropout=0.2,
        glu=True,
        activation="glu",
        pooling_type="mean",
        kernel_sizes=[3, 5],
    )

    model2 = MaskedLengthPredictionModule(
        embed_dim=C,
        length_hidden_dim=128,
        max_target_positions=5,
        length_dropout=0.2,
    )

    model3 = MaskedLengthRegressionModule(
        embed_dim=C,
        length_hidden_dim=128,
        length_dropout=0.2,
    )

    log_probs = model(x, padding_mask)
    log_probs2 = model2(x, padding_mask)
    regression = model3(x, padding_mask)
    # parameter count
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"{total_params=:,}")
    print(f"{log_probs=}")
    
    print(f"{torch.argmax(log_probs2, dim=-1)=}")
    print(f"{log_probs2=}")
    print(f"{log_probs2.shape=}")
    print(f"{regression=}")
