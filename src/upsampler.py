import torch
import torch.nn as nn


class ConvUpsample(nn.Module):
    def __init__(self, d_model, upsample_factor):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.upsample_factor,
            stride=1,
            padding=self.upsample_factor // 2,
            groups=d_model,   # depthwise
        )

    def forward(self, x, masks=None):
        # x: B x T x D
        x = x.transpose(1, 2)        # B x D x T
        x = self.conv(x)             # B x D x T
        x = x.repeat_interleave(self.upsample_factor, dim=-1)
        if masks is None:
            return x.transpose(1, 2), None
        else:
            masks = masks.repeat_interleave(self.upsample_factor, dim=-1)

        return x.transpose(1, 2), masks     # B x (T*upsample) x D


class LearnedUpsample(nn.Module):
    def __init__(self, d_model, upsample_factor):
        super().__init__()
        self.upsample_factor = upsample_factor
        # Transposed Convolution이 '역'으로 연산하여 길이를 늘려줍니다.
        self.conv_t = nn.ConvTranspose1d(
            d_model,
            d_model,
            kernel_size=upsample_factor,
            stride=upsample_factor  # stride가 곧 배수(factor)가 됩니다.
        )

    def forward(self, x, masks=None):
        # x: B x T x D
        x = x.transpose(1, 2)       # B x D x T
        x = self.conv_t(x)          # B x D x (T * factor) (알아서 늘어남)

        if masks is None:
            return x.transpose(1, 2), None
        # masks: B x T
        if masks is not None:            
            masks = masks.repeat_interleave(self.upsample_factor, dim=-1)            
        
        return x.transpose(1, 2), masks


if __name__ == "__main__":
    B, T, D = 2, 5, 4
    # masks
    lengths = torch.tensor([3, 5])
    masks = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(-1)
    upsample_factor = 3
    x = torch.randn(B, T, D)
    upsampler = LearnedUpsample(D, upsample_factor)
    y, y_masks = upsampler(x, masks=masks)
    print(f"Upsample factor: {upsample_factor}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # Expected: (B, T * upsample_factor, D)
    print(f"Masks: {y_masks}")
