import torch
import torch.nn as nn
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

class AdaLN(nn.Module):
    """
    Adaptive Layer Norm (FiLM 메커니즘 적용)
    Time Embedding(cond)을 받아 Scale(gamma)과 Shift(beta)를 예측하여 적용
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x, cond):
        # cond: Timestep Embedding
        # shift, scale 예측 (Broadcasting을 위해 차원 조절)
        shift, scale = self.emb(cond).chunk(2, dim=1)
        shift = shift.unsqueeze(1) # (Batch, 1, Dim)
        scale = scale.unsqueeze(1) # (Batch, 1, Dim)
        
        # FiLM 연산: (1 + scale) * norm(x) + shift
        return self.norm(x) * (1 + scale) + shift

class DiTBlock(nn.Module):
    """
    DiT Block with Multi-Modal Cross Attention
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLN(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = AdaLN(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm3 = AdaLN(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )

    def forward(self, x, t_emb, context, context_mask=None):
        # 1. Self-Attention (x끼리의 관계)
        x = x + self.attn(self.norm1(x, t_emb), self.norm1(x, t_emb), self.norm1(x, t_emb))[0]
        
        # 2. Cross-Attention (x와 Audio+Text context 간의 관계)
        # Query: x (Main Modality)
        # Key, Value: context (Audio + Text Concatenated)
        x_norm = self.norm2(x, t_emb)
        x = x + self.cross_attn(query=x_norm, key=context, value=context, key_padding_mask=context_mask)[0]
        
        # 3. MLP (Feed Forward)
        x = x + self.mlp(self.norm3(x, t_emb))
        return x

  
class DiscreteMultiModalDiT(nn.Module):
    def __init__(
        self,
        vocab_size=10000,   # Discrete Target의 Vocab 크기
        hidden_size=512,
        depth=6,
        num_heads=8,
        audio_dim=80,       # Audio Feature 차원 (예: Mel-spec)
        text_dim=768        # Text Embedding 차원 (예: BERT output)
    ):
        super().__init__()
        
        # 1. Input Embedding (Discrete Tokens -> Vector)
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, hidden_size)) # Positional Embedding
        
        # 2. Time Embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # 3. Condition Projectors (차원 맞추기용)
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 5. Output Head (Vector -> Logits)
        self.final_norm = AdaLN(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Xavier init etc.
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x_t, t, audio_feats, text_feats, audio_mask=None, text_mask=None):
        """
        x_t: (Batch, Seq_Len) - Noisy Discrete Token Indices
        t: (Batch,) - Time steps
        audio_feats: (Batch, Audio_Len, Audio_Dim)
        text_feats: (Batch, Text_Len, Text_Dim)
        """
        
        # 1. Embeddings
        x = self.token_emb(x_t) + self.pos_emb[:, :x_t.size(1), :]
        t_emb = self.t_embedder(t)
        
        # 2. Context Fusion (Concatenation)
        # 서로 다른 길이의 Audio와 Text를 Hidden Dimension으로 투영 후 시간축으로 이어 붙임
        proj_audio = self.audio_proj(audio_feats) # (B, A_Len, H)
        proj_text = self.text_proj(text_feats)    # (B, T_Len, H)
        
        context = torch.cat([proj_audio, proj_text], dim=1) # (B, A_Len + T_Len, H)
        
        # Mask 처리 (필요 시 Padding 무시를 위해 결합)
        context_mask = None
        if audio_mask is not None and text_mask is not None:
             context_mask = torch.cat([audio_mask, text_mask], dim=1)
        
        # 3. DiT Blocks Pass
        for block in self.blocks:
            x = block(x, t_emb, context, context_mask)
            
        # 4. Final Head
        x = self.final_norm(x, t_emb)
        logits = self.head(x) # (Batch, Seq_Len, Vocab_Size)
        
        return logits
    

if __name__ == "__main__":
    # 모델 초기화
    model = DiscreteMultiModalDiT(
        vocab_size=5000,
        hidden_size=256,
        audio_dim=80,  # Mel-spectrogram
        text_dim=768   # BERT output size
    ).cuda()

    # trainable 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    # 더미 데이터 (Batch=2)
    x_t = torch.randint(0, 5000, (2, 128)).cuda()    # 128 길이의 타겟 시퀀스
    t = torch.tensor([0.1, 0.9]).cuda()               # 임의의 타임스텝
    audio = torch.randn(2, 500, 80).cuda()           # 500 프레임 오디오
    text = torch.randn(2, 30, 768).cuda()            # 30 토큰 텍스트

    # Forward
    logits = model(x_t, t, audio, text)
    print(logits.shape) 
    # 출력: torch.Size([2, 128, 5000]) -> (Batch, Seq, Vocab)