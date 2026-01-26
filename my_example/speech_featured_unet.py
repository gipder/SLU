import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DiscreteContextUnetConfig:
    num_classes: int
    n_feat: int = 256
    ctx_dim: int = 768
    n_heads: int = 8

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.shortcut = nn.Identity()
        if not self.same_channels and is_res:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            out = self.shortcut(x) + x2
            #if self.same_channels:
            #    out = x + x2
            #else:
            #    out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class CrossAttention2D(nn.Module):
    """
    x:       [B, C, H, W]   (image features, queries)
    context: [B, T, D_ctx]  (text or other sequence features)
    mask:    [B, T]         (1: valid, 0: pad) or None
    """
    def __init__(self, dim: int, context_dim: int, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context, context_mask=None):
        B, C, H, W = x.shape
        N = H * W

        # [B, C, H, W] -> [B, N, C]
        x_tokens = x.view(B, C, N).transpose(1, 2)  # [B, N, C]

        # Q, K, V
        q = self.to_q(x_tokens)        # [B, N, C]
        k = self.to_k(context)         # [B, T, C]
        v = self.to_v(context)         # [B, T, C]

        # [B, L, C] -> [B, heads, L, head_dim]
        def split_heads(t):
            B, L, C = t.shape
            t = t.view(B, L, self.n_heads, self.head_dim)
            return t.permute(0, 2, 1, 3)  # [B, H, L, D_h]

        q = split_heads(q)   # [B, H, N, D_h]
        k = split_heads(k)   # [B, H, T, D_h]
        v = split_heads(v)   # [B, H, T, D_h]

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, T]

        if context_mask is not None:
            # mask: [B, T] (1: valid, 0: pad) -> True=mask out
            mask = (context_mask == 0).unsqueeze(1).unsqueeze(1)  # [B,1,1,T]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)  # [B, H, N, T]
        out = torch.matmul(attn, v)                # [B, H, N, D_h]

        # [B, H, N, D_h] -> [B, N, C]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.to_out(out)                     # [B, N, C]

        # [B, N, C] -> [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class DiscreteContextUnet(nn.Module):
    #def __init__(self, num_classes, n_feat=256, ctx_dim=768, n_heads=8):
    def __init__(self, cfg: DiscreteContextUnetConfig):
        """
        num_classes: number of class (K)
        n_feat:      base channels
        ctx_dim:     context feature dim (ex: 텍스트 인코더 출력 차원)
        """
        super(DiscreteContextUnet, self).__init__()

        self.in_channels = cfg.num_classes
        self.n_feat = cfg.n_feat
        self.ctx_dim = cfg.ctx_dim

        self.init_conv = ResidualConvBlock(cfg.num_classes, cfg.n_feat, is_res=True)

        self.down1 = UnetDown(cfg.n_feat, cfg.n_feat)
        self.down2 = UnetDown(cfg.n_feat, 2 * cfg.n_feat)

        # 28x28 기준으로 7x7 -> 1x1
        self.to_vec = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.GELU()
        )
        #self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # time embedding은 그대로 사용
        self.timeembed1 = EmbedFC(1, 2 * cfg.n_feat)
        self.timeembed2 = EmbedFC(1, 1 * cfg.n_feat)

        # cross-attention 모듈 (bottleneck: 2*n_feat, 중간: n_feat)
        self.ctx_attn_mid = CrossAttention2D(dim=2 * cfg.n_feat, context_dim=cfg.ctx_dim, n_heads=cfg.n_heads)
        self.ctx_attn_mid2 = CrossAttention2D(dim=cfg.n_feat, context_dim=cfg.ctx_dim, n_heads=cfg.n_heads)

        self.up0_conv = nn.Sequential(
            nn.Conv2d(2*cfg.n_feat, 2*cfg.n_feat, 3, 1, 1),
            nn.GroupNorm(8, 2*cfg.n_feat),
            nn.ReLU()
        )
        """
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        """
        self.up1 = UnetUp(4 * cfg.n_feat, cfg.n_feat)
        self.up2 = UnetUp(2 * cfg.n_feat, cfg.n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * cfg.n_feat, cfg.n_feat, 3, 1, 1),
            nn.GroupNorm(8, cfg.n_feat),
            nn.ReLU(),
            nn.Conv2d(cfg.n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, context, context_mask=None):
        """
        x:      [B, K, H, W]  (discrete noisy image)
        t:      [B] or [B,1]  (time)        
        context:       [B, T, ctx_dim],
        context_mask:  [B, T] (optional, 1: valid, 0: pad)
        """
        B = x.shape[0]
        device = x.device

        context = context.to(device)               # [B, T, ctx_dim]
        context_mask = context_mask
        if context_mask is not None:
            context_mask = context_mask.to(device)

        # encoder
        x0 = self.init_conv(x.float())                              # [B, n_feat, H, W]
        down1 = self.down1(x0)                              # [B, n_feat, H/2, W/2]
        down2 = self.down2(down1)                           # [B, 2*n_feat, H/4, W/4]
        hiddenvec = self.to_vec(down2)                      # [B, 2*n_feat, 1, 1]

        # time embedding
        if t.dim() == 1:
            t_in = t.view(B, 1)
        else:
            t_in = t                                        # [B,1] 이라고 가정

        temb1 = self.timeembed1(t_in).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t_in).view(-1, self.n_feat, 1, 1)

        # --- bottleneck: time + cross-attention ---

        # time 먼저 더해주고
        h_mid = hiddenvec + temb1                           # [B, 2*n_feat, 1, 1]

        # context와 cross-attn (단, spatial이 1x1이라 토큰 하나에 cross-attn 하는 형태)
        h_mid_attn = self.ctx_attn_mid(h_mid, context, context_mask) + h_mid

        target_h, target_w = down2.shape[2], down2.shape[3]
        up1_input = h_mid_attn.expand(-1, -1, target_h, target_w) 
        # up0로 업샘플
        up1 = self.up0_conv(up1_input)      # [B, 2n, H/4, W/4]
        
        # up1+down2를 UnetUp에 보냄
        up2 = self.up1(up1, down2)                          # [B, n_feat, H/2, W/2]

        # 여기에 temb2 더하고, 다시 cross-attention
        h_mid2 = up2 + temb2                                # [B, n_feat, H/2, W/2]
        h_mid2_attn = self.ctx_attn_mid2(h_mid2, context, context_mask) + h_mid2

        # 마지막 up + output
        up3 = self.up2(h_mid2_attn, down1)                  # [B, n_feat, H, W]
        out = self.out(torch.cat((up3, x0), 1))             # [B, in_channels, H, W]

        return out
    
if __name__ == "__main__":

    B = 2
    H = 4
    W = 4
    C = 1
    K = 650
    D = 512
    n_H = 4
    T = 10

    cfg = DiscreteContextUnetConfig(
        num_classes=K,
        n_feat=D,
        ctx_dim=D,
        n_heads=n_H
    )
    
    xt = torch.randint(0, K, (B, H, W))
    t = torch.rand((B,))
    xt = nn.functional.one_hot(xt, K)  # Noisy input at time t (B, H, W, K)
    xt = xt.permute(0, 3, 1, 2) # B, K, H, W
    feat = torch.rand((B, T, D))
    feat_mask = torch.ones((B, T))
    unet = DiscreteContextUnet(cfg)

    logits = unet(xt, t, feat, feat_mask)
    print(f"{logits.shape=}")


    
    

