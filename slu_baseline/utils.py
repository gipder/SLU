import torch
import numpy as np
import imageio.v2 as imageio
from einops import rearrange
import random
import argparse
import logging
import os
import time
from jiwer import process_words, process_characters


logger = logging.getLogger(__name__)


def setup_logger(save_dir, log_name="train"):
    """Setup logger that writes to both console and file"""
    # Root logger를 설정해서 모든 모듈의 logger가 이 핸들러를 사용하도록 함
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    
    # Shared formatter for both console and file (파일이름 포함)
    formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f"{log_name}-{time.strftime('%Y%m%d-%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to {log_file}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys (for DataParallel/DDP)"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

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

def compute_wer_cer(text_hyps_list, text_targets_list):
    """
    Compute WER (Word Error Rate) and CER (Character Error Rate) using jiwer.
    
    Args:
        text_hyps_list: List of hypothesis string lists
        text_targets_list: List of target string lists
                
    Returns:
        Dictionary containing:
            - wer: Word Error Rate (0-1)
            - cer: Character Error Rate (0-1)
            - num_sentences: Total number of sentences
            - wer_details: Dict with 'hits', 'substitutions', 'deletions', 'insertions'
            - cer_details: Dict with 'hits', 'substitutions', 'deletions', 'insertions'
    """
    
    assert len(text_hyps_list) == len(text_targets_list), "Length mismatch between hyps and targets"
    # Compute detailed metrics using jiwer.compute_measures
    word_measures = process_words(text_targets_list, text_hyps_list)
    character_measures = process_characters(text_targets_list, text_hyps_list)    
    
    results = {
        'num_sentences': len(text_targets_list),
        'wer': word_measures.wer,
        'cer': character_measures.cer,
        'wer_details': {            
            'hits': word_measures.hits,
            'substitutions': word_measures.substitutions,
            'deletions': word_measures.deletions,
            'insertions': word_measures.insertions,
        },
        'cer_details': {            
            'hits': character_measures.hits,
            'substitutions': character_measures.substitutions,
            'deletions': character_measures.deletions,
            'insertions': character_measures.insertions,
        }
    }

    return results


def class_name(obj) -> str:
    """
    Docstring for class_name
    
    :param obj: Description
    :return: Description
    :rtype: str
    """
    return type(obj).__name__