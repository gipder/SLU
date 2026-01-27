import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Optional
from tqdm import tqdm

# For DFML
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver

def sampling(
    model: ModelWrapper,                
    audio_feats: torch.Tensor,
    audio_mask: torch.Tensor,
    text_feats: torch.Tensor,
    text_mask: torch.Tensor,        
    n_step: int,
    K: int,    
    mask_id: int,        
    return_intermediates: Optional[bool] = True,
    is_uniform: Optional[bool] = False,
    step_size: Optional[float] = 0.01,
    eps: Optional[float] = 1e-4,
    device: Optional[torch.DeviceObjType] = "cuda",
    oracle_lengths: Optional[torch.Tensor] = None,
    
) -> torch.Tensor:
    """
    Docstring for sampling
    
    :param model: Description
    :type model: ModelWrapper
    :param audio_feats: Description
    :type audio_feats: torch.Tensor
    :param audio_mask: Description
    :type audio_mask: torch.Tensor
    :param text_feats: Description
    :type text_feats: torch.Tensor
    :param text_mask: Description
    :type text_mask: torch.Tensor
    :param n_step: Description
    :type n_step: int
    :param K: Description
    :type K: int
    :param max_output_length: Description
    :type max_output_length: int
    :param mask_id: Description
    :type mask_id: int
    :param return_intermediates: Description
    :type return_intermediates: Optional[bool]
    :param is_uniform: Description
    :type is_uniform: Optional[bool]
    :param step_size: Description
    :type step_size: Optional[float]
    :param eps: Description
    :type eps: Optional[float]
    :param device: Description
    :type device: Optional[torch.DeviceObjType]
    :return: Description
    :rtype: Tensor
    """

    model = model

    if model.training is False:
        model.eval()

    # a convex path path
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)

    solver = MixtureDiscreteEulerSolver(
        model=model,
        path=path,
        vocabulary_size=K,
    )
    
    step_size = step_size
    time_grid = torch.linspace(0.0, 1.0 - eps, n_step, device=device)

    B = audio_feats.size(0) 
    if oracle_lengths is not None:
        T = oracle_lengths.max().item()
    else:   
        # normal case: predict lengths
        pred_lengths = model.get_length_logits(text_feats, text_mask.bool())
        T = torch.argmax(pred_lengths, dim=-1).max().item()    

    if is_uniform:
        x_0 = torch.randint(0, K, (B, T), device=device)
    else:
        x_0 = torch.full((B, T), mask_id, device=device)
    
    x_1_hat = solver.sample(
        x_init=x_0,
        step_size=step_size,
        time_grid=time_grid,
        return_intermediates=return_intermediates,            
        audio_feats=audio_feats,
        audio_mask=audio_mask,
        text_feats=text_feats,
        text_mask=text_mask,
    )

    return x_1_hat


def sampling_batch(
    test_dl: DataLoader,    
    model: ModelWrapper,    
    n_step: int,
    K: int,    
    mask_id: int,        
    return_intermediates: Optional[bool] = True,
    is_uniform: Optional[bool] = False,
    step_size: Optional[float] = 0.01,
    eps: Optional[float] = 1e-4,
    device: Optional[torch.DeviceObjType] = "cuda",
    verbose: Optional[bool] = False,
    use_oracle_length: Optional[bool] = False,
):
    """
    Docstring for sampling_batch
    
    :param test_dl: Description
    :type test_dl: DataLoader
    :param model: Description
    :type model: ModelWrapper
    :param fusion_model: Description
    :type fusion_model: nn.Module
    :param n_step: Description
    :type n_step: int
    :param K: Description
    :type K: int
    :param max_output_length: Description
    :type max_output_length: int
    :param mask_id: Description
    :type mask_id: int
    :param return_intermediates: Description
    :type return_intermediates: Optional[bool]
    :param is_uniform: Description
    :type is_uniform: Optional[bool]
    :param step_size: Description
    :type step_size: Optional[float]
    :param eps: Description
    :type eps: Optional[float]
    :param device: Description
    :type device: Optional[torch.DeviceObjType]
    :param verbose: Description
    :type verbose: Optional[bool]
    """
    model.eval()    

    hyp = []
    target = []

    if verbose:
        dataloader = tqdm(test_dl)
    else:
        dataloader = test_dl

    for batch in dataloader:
        (
            audio_feats, audio_feat_mask,
            text_feats, text_feat_mask,
            x_1, x_1_mask 
        ) = batch

        audio_feats = audio_feats.to(device)
        audio_feat_mask = audio_feat_mask.to(device)
        text_feats = text_feats.to(device)
        text_feat_mask = text_feat_mask.to(device)
        
        if use_oracle_length:
            oracle_lengths = x_1_mask.sum(dim=-1)
        else:
            oracle_lengths = None

        x_1_hat = sampling(
            model=model,  
            audio_feats=audio_feats,
            audio_mask=audio_feat_mask,
            text_feats=text_feats,
            text_mask=text_feat_mask,                      
            n_step=n_step,
            K=K,            
            mask_id=mask_id,            
            return_intermediates=return_intermediates,
            is_uniform=is_uniform,
            step_size=step_size,
            eps=eps,
            device=device,
            oracle_lengths=oracle_lengths,            
        )

        if verbose:
            print(f"{x_1=}")
            print(f"{x_1_hat[-1]=}")

        target.append(x_1)
        hyp.append(x_1_hat[-1])        

    final_hyp = torch.cat(hyp, dim=0).tolist()
    final_target = torch.cat(target, dim=0).tolist()    

    return final_hyp, final_target

def sampling_debugging(
    test_dl: DataLoader,
    model: ModelWrapper,    
    n_step: int,
    K: int,    
    mask_id: int,
    return_intermediates: Optional[bool] = True,
    is_uniform: Optional[bool] = False,
    step_size: Optional[float] = 0.01,
    eps: Optional[float] = 1e-4,
    device: Optional[torch.DeviceObjType] = "cuda",    
    verbose: Optional[bool] = False,
    use_oracle_length: Optional[bool] = False,
):
    """
    Docstring for sampling_debugging

    :param test_dl: Description
    :type test_dl: DataLoader
    :param model: Description
    :type model: ModelWrapper
    :param n_step: Description
    :type n_step: int
    :param K: Description
    :type K: int
    :param max_output_length: Description
    :type max_output_length: int
    :param mask_id: Description
    :type mask_id: int
    :param return_intermediates: Description
    :type return_intermediates: Optional[bool]
    :param is_uniform: Description
    :type is_uniform: Optional[bool]
    :param step_size: Description
    :type step_size: Optional[float]
    :param eps: Description
    :type eps: Optional[float]
    :param device: Description
    :type device: Optional[torch.DeviceObjType]
    :param verbose: Description
    :type verbose: Optional[bool]
    """
    model.eval()    
    
    hyp = []
    target = []
    
    batch = next(iter(test_dl))
    (
        audio_feats, audio_feat_mask,
        text_feats, text_feat_mask,
        gts, hyps,
        gt_mask, hyp_mask,
        str_gts, str_hyps,        
    ) = batch

    audio_feats = audio_feats.to(device)
    audio_feat_mask = audio_feat_mask.to(device)
    
    text_feats = text_feats.to(device)
    text_feat_mask = text_feat_mask.to(device)

    if use_oracle_length:
        oracle_lengths = gt_mask.sum(dim=-1)
    else:
        oracle_lengths = None

    x_1 = gts.to(device)
    
    x_1_hat = sampling(
        model=model,            
        audio_feats=audio_feats,
        audio_mask=audio_feat_mask,
        text_feats=text_feats,
        text_mask=text_feat_mask,
        n_step=n_step,
        K=K,        
        mask_id=mask_id,            
        return_intermediates=return_intermediates,
        is_uniform=is_uniform,
        step_size=step_size,
        eps=eps,
        device=device, 
        oracle_lengths=oracle_lengths,           
    )

    # S, B, T -> B, S, T
    x_1_hat = x_1_hat.permute(1, 0, 2)
    
    hyp = []
    target = []
    asr_hyp = []
    for b in range(x_1_hat.size(0)):
        for s in range(x_1_hat.size(1)):
            ids = x_1_hat[b, s].tolist()
            sentence = ids        
        target.append(gts[b].tolist())
        hyp.append(sentence)
        asr_hyp.append(hyps[b])

    return hyp, target, asr_hyp, str_gts
