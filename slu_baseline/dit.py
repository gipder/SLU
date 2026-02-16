import torch
import torch.nn as nn
import math

# (이전 답변의 TimestepEmbedder, AdaLN 클래스가 여기에 포함되어 있다고 가정합니다)
# 필요하다면 다시 적어드릴 수 있습니다.
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
        nn.init.constant_(self.emb[-1].weight, 0)
        nn.init.constant_(self.emb[-1].bias, 0)

    def forward(self, x, cond):
        # cond: Timestep Embedding
        # shift, scale 예측 (Broadcasting을 위해 차원 조절)
        shift, scale = self.emb(cond).chunk(2, dim=1)
        shift = shift.unsqueeze(1) # (Batch, 1, Dim)
        scale = scale.unsqueeze(1) # (Batch, 1, Dim)
        
        # FiLM 연산: (1 + scale) * norm(x) + shift
        return self.norm(x) * (1 + scale) + shift
    
class DualConditionDiTBlock(nn.Module):
    """
    DiT Block with Separated Cross-Attention Layers
    Audio와 Text를 위한 Cross-Attention을 분리하여 순차적으로 수행
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        # 1. Self-Attention
        self.norm1 = AdaLN(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 2. Audio Cross-Attention
        self.norm2_audio = AdaLN(hidden_size)
        self.cross_attn_audio = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 3. Text Cross-Attention
        self.norm2_text = AdaLN(hidden_size)
        self.cross_attn_text = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        # 4. Feed Forward
        self.norm3 = AdaLN(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )

    def forward(self, x, t_emb, audio_feats, text_feats, audio_mask=None, text_mask=None):
        """
        x: (Batch, Seq_Len, Dim) - Main Modality
        t_emb: (Batch, Dim) - Time Embedding
        audio_feats: (Batch, Audio_Len, Dim) - Condition 1
        text_feats: (Batch, Text_Len, Dim) - Condition 2
        """
        
        # 1. Self-Attention (Global Context)
        # FiLM(AdaLN)으로 Time step 정보 주입 후 Self-Attn
        x_norm = self.norm1(x, t_emb)
        x = x + self.attn(query=x_norm, key=x_norm, value=x_norm)[0]
        
        # 2. Audio Cross-Attention (Look at Audio)
        # 쿼리(Q)는 현재 x, 키(K)/값(V)은 Audio
        x_norm = self.norm2_audio(x, t_emb)
        x = x + self.cross_attn_audio(
            query=x_norm, 
            key=audio_feats, 
            value=audio_feats, 
            key_padding_mask=audio_mask
        )[0]
        
        # 3. Text Cross-Attention (Look at Text)
        # 쿼리(Q)는 Audio 정보를 머금은 x, 키(K)/값(V)은 Text
        x_norm = self.norm2_text(x, t_emb)
        x = x + self.cross_attn_text(
            query=x_norm, 
            key=text_feats, 
            value=text_feats, 
            key_padding_mask=text_mask
        )[0]
        
        # 4. MLP
        x_norm = self.norm3(x, t_emb)
        x = x + self.mlp(x_norm)
        
        return x

class DiscreteDualDiT(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=512,
        depth=6,
        num_heads=8,
        audio_dim=80,
        text_dim=768
    ):
        super().__init__()
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Projectors (차원 매칭용)
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        
        # Blocks (DualConditionBlock 사용)
        self.blocks = nn.ModuleList([
            DualConditionDiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # Final Head
        self.final_norm = AdaLN(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        # 추가적인 weight 초기화 로직...

    def forward(self, x_t, t, audio_feats, text_feats, audio_mask=None, text_mask=None):
        # 1. Embedding
        # x_t: (Batch, Seq_Len)
        # t: (Batch,)
        # audio_feats: (Batch, Audio_Len, Audio_Dim)
        # text_feats: (Batch, Text_Len, Text_Dim)
        # audio_mask: (Batch, Audio_Len)
        # text_mask: (Batch, Text_Len)
        x = self.token_emb(x_t) + self.pos_emb[:, :x_t.size(1), :]
        t_emb = self.t_embedder(t)
        
        # 2. Condition Projection (Concat 하지 않음!)
        proj_audio = self.audio_proj(audio_feats) # (B, Audio_Len, H)
        proj_text = self.text_proj(text_feats)    # (B, Text_Len, H)
        
        # 3. Blocks Pass (Audio와 Text를 별도 인자로 전달)
        for block in self.blocks:
            x = block(x, t_emb, proj_audio, proj_text, audio_mask, text_mask)
            
        # 4. Output
        x = self.final_norm(x, t_emb)
        logits = self.head(x)
                
        return logits
    

if __name__ == "__main__":
    # 모델 초기화
    model = DiscreteDualDiT(
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