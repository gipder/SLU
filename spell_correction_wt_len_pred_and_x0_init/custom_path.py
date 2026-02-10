import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver
from model import DFMModel, DFMModelConfig, DFMModelWrapper

class UniformDiscreteProbPath(MixtureDiscreteProbPath):
    def __init__(self, scheduler, vocab_size, noise_ratio=0.5):
        super().__init__(scheduler=scheduler)
        self.vocab_size = vocab_size
        self.noise_ratio = noise_ratio

    def sample(self, t, x_0, x_1):
        # 1. 기존 Mixture 방식으로 확률 계산 (Straight line)
        # 보통 부모 클래스에 compute_prob 같은 메서드가 있다면 그걸 활용하지만,
        # 여기서는 원리를 보여드리기 위해 직접 계산 로직을 섞습니다.
        # t의 모양을 맞춤
        t_b = t.view(-1, 1) 
        
        # 2. Uniform Noise 확률 (모든 토큰이 나올 확률 = 1/vocab_size)
        # 실제 구현에서는 log_prob를 쓸 수도 있고 prob를 쓸 수도 있습니다.
        # 여기서는 개념적으로 설명합니다.
        
        # 우리가 원하는 궤적:
        # P_t = (1 - alpha_t) * Mixture(x0, x1) + alpha_t * Uniform
        
        # alpha_t는 t=0, t=1일 때는 0이어야 하고, 중간(t=0.5)에서 커져야 합니다.
        # 가장 간단한 형태: alpha_t = 4 * t * (1-t) * noise_ratio
        alpha_t = 4 * t_b * (1 - t_b) * self.noise_ratio        
        # 실제로는 여기서 부모 클래스의 sample 로직을 호출한 뒤,
        # 일정 확률(alpha_t)로 랜덤 토큰으로 덮어쓰는 방식이 가장 구현하기 쉽습니다.
        
        # A. 원래 경로 샘플링 (x0 -> x1)        
        original_sample = super().sample(t=t, x_0=x_0, x_1=x_1)
        x_t = original_sample.x_t
        
        # B. 노이즈 마스크 생성 (alpha_t 확률로 True)
        # alpha_t 확률만큼은 Uniform(랜덤 토큰)에서 가져옴
        noise_mask = torch.rand_like(x_t.float()) < alpha_t
        
        # C. 랜덤 토큰 생성
        random_tokens = torch.randint(0, self.vocab_size, x_t.shape, device=x_t.device)
        
        # D. 섞어주기
        x_t_new = torch.where(noise_mask, random_tokens, x_t)
        
        # 리턴 객체 업데이트 (x_t 교체)
        original_sample.x_t = x_t_new
        return original_sample

if __name__ == "__main__":
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = UniformDiscreteProbPath(scheduler=scheduler, vocab_size=10, noise_ratio=0.5)

    device = "cuda"
    # sample time t ~ Uniform(eps,1)
    t = torch.rand((1,), device=device).clamp(1e-4, 1.0 - 1e-4)
    t = torch.tensor([0.5 ], device=device)  # 예시
    print(f"{t=}")

    x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], device=device)  # shape: (B,)
    x0 = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 8, 0, 9, 0, 0, 0, 0, 0]], device=device)  # shape: (B,)
    #x0 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device=device)  # shape: (B,)

    print(f"{t.shape=}")
    sample = path.sample(t=t, x_0=x0, x_1=x1)
    print(f"{sample.t=}")
    print(f"{sample.x_0=}")
    print(f"{sample.x_t=}")
    print(f"{sample.x_1=}")

    B = 2
    K = 650
    T_out = 16
    D = 1024
    n_H = 8
    K=43

    cfg = DFMModelConfig(
        vocab_size=K,
        hidden_size=D,
        audio_dim=D,
        text_dim=D,
        num_heads=n_H
    )

    print(f"{cfg=}")
    model = DFMModel(cfg)

    audio_feats = torch.rand((B, T_out * 4, D)).to(device)
    audio_mask = torch.ones(B, T_out * 4).bool().to(device)

    text_feats = torch.rand((B, T_out * 2, D)).to(device)
    text_mask = torch.ones(B, T_out * 2).bool().to(device)  
    eps = 1e-4
    n_step = 10
    time_grid = torch.linspace(0.0, 1.0 - eps, n_step, device=device)
    x_0 = torch.randint(0, K, (B, T_out), device=device)

    probability_denoiser = DFMModelWrapper(model).to(device)

    solver = MixtureDiscreteEulerSolver(
            model=probability_denoiser,
            path=path,
            vocabulary_size=K,
        )

    x_1_hat = solver.sample(
            x_init=x_0,
            step_size=0.1,
            time_grid=time_grid,
            return_intermediates=True,            
            audio_feats=audio_feats,
            audio_mask=audio_mask,
            text_feats=text_feats,
            text_mask=text_mask,
    )
    print(f"{x_0=}")
    print(f"{x_1_hat=}")