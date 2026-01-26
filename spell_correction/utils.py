import torch
import numpy as np
import imageio.v2 as imageio
from einops import rearrange
import random


"""
def make_gif_from_xts(xts, n_plots=10, out_path="out.gif", fps=8):
    # 샘플링 (기존 로직 그대로)
    stride = max(1, len(xts) // n_plots)
    xts_sel = [xts[i * stride] for i in range(n_plots)] + [xts[-1]]

    frames = []
    for xt in xts_sel:
        # xt: [B,H,W]
        grid = torch.clamp(xt - 1, 0).cpu().numpy()  # (B,H,W)
        grid = rearrange(grid, 'b h w -> h (b w)')    # 한 프레임에 B개를 가로로 붙임

        # 0~1 또는 0~255로 맞춰서 uint8로 변환
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        frame = (grid * 255).astype(np.uint8)         # [H, B*W]
        frames.append(frame)

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path
"""

def set_seed(seed: int):
    """
    Set random seed for reproducibility
    Args:
        seed (int): Random seed value
    """

    # 1. 난수 시드 고정 (속도 저하 없음, 실험 재현성 필수)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. cuDNN 설정 변경 (속도 최적화 모드)
    # deterministic = True는 속도를 저하시키므로 제거 (기본값이 False)
    # benchmark = True로 설정하여 하드웨어에 가장 빠른 알고리즘을 자동 선택하게 함
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):    
    """
    Set seed for each data loader worker
    Args:
        worker_id (int): Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_gif_from_xts(
    xts,
    n_plots=10,
    out_path="out.gif",
    fps=8,
    mode="animate",          # "animate" or "stack"
    normalize="per_frame",   # "per_frame" or "global"
):
    # 샘플링
    """
    stride = max(1, len(xts) // n_plots)    
    xts_sel = [xts[i * stride] for i in range(n_plots)] + [xts[-1]]
    """    
    idxs = torch.linspace(
        0, len(xts) - 1,
        steps=min(n_plots + 1, len(xts))
    ).long()
    xts_sel = [xts[i] for i in idxs]

    # (선택) 전체 프레임에 대해 동일 스케일로 정규화
    if normalize == "global":
        mins, maxs = [], []
        for xt in xts_sel:
            g = torch.clamp(xt - 1, 0).float()
            mins.append(g.min().item())
            maxs.append(g.max().item())
        gmin, gmax = float(min(mins)), float(max(maxs))
    else:
        gmin = gmax = None

    def xt_to_frame(xt):
        # xt: [B,H,W]
        grid = torch.clamp(xt - 1, 0).float().cpu().numpy()  # (B,H,W)
        grid = rearrange(grid, 'b h w -> h (b w)')           # [H, B*W]

        if normalize == "global":
            grid_n = (grid - gmin) / (gmax - gmin + 1e-8)
        else:
            grid_n = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

        frame = (grid_n * 255).astype(np.uint8)              # [H, B*W]
        return frame

    frames = [xt_to_frame(xt) for xt in xts_sel]

    if mode == "animate":
        imageio.mimsave(out_path, frames, fps=fps)
        return out_path

    if mode == "stack":
        # 시간축을 세로로 붙여서 한 장으로
        stacked = np.concatenate(frames, axis=0)  # [T*H, B*W]
        # GIF로 저장하면 1프레임짜리 GIF (원하면 PNG로 저장도 가능)
        if out_path.lower().endswith(".gif"):
            imageio.mimsave(out_path, [stacked], fps=fps)
        else:
            imageio.imwrite(out_path, stacked)
        return out_path

    raise ValueError("mode must be 'animate' or 'stack'")

