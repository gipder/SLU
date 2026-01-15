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
    input: torch.Tensor,
    input_mask: torch.Tensor,        
    n_step: int,
    K: int,
    max_output_length: int,
    mask_id: int,        
    return_intermediates: Optional[bool] = True,
    is_uniform: Optional[bool] = False,
    step_size: Optional[float] = 0.01,
    eps: Optional[float] = 1e-4,
    device: Optional[torch.DeviceObjType] = "cuda",
) -> torch.Tensor:
    """
    Docstring for sampling
    
    :param model: Description
    :type model: ModelWrapper    
    :param input: Description
    :type input: torch.Tensor
    :param input_mask: Description
    :type input_mask: torch.Tensor
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

    B = input.size(0)
    T = max_output_length

    if is_uniform:
        x_0 = torch.randint(0, K, (B, T), device=device)
    else:
        x_0 = torch.full((B, T), mask_id, device=device)
    
    x_1_hat = solver.sample(
        x_init=x_0,
        step_size=step_size,
        time_grid=time_grid,
        return_intermediates=return_intermediates,            
        emb_seq=input,
        emb_mask=input_mask,
    )

    return x_1_hat


def sampling_batch(
    test_dl: DataLoader,    
    model: ModelWrapper,    
    n_step: int,
    K: int,
    max_output_length: int,
    mask_id: int,        
    return_intermediates: Optional[bool] = True,
    is_uniform: Optional[bool] = False,
    step_size: Optional[float] = 0.01,
    eps: Optional[float] = 1e-4,
    device: Optional[torch.DeviceObjType] = "cuda",
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
    """
    model.eval()    

    hyp = []
    target = []
    for batch in tqdm(test_dl):
        (
            audio_feats, audio_feat_mask,
            text_feats, text_feat_mask,
            x_1, x_1_mask 
        ) = batch

        audio_feats = audio_feats.to(device)
        audio_feat_mask = audio_feat_mask.to(device)
        text_feats = text_feats.to(device)
        text_feat_mask = text_feat_mask.to(device)

        B = audio_feats.size(0)
        T = max_output_length        

        # audio feature only
        emb_seq = audio_feats
        emb_mask = audio_feat_mask

        x_1_hat = sampling(
            model=model,            
            input=emb_seq,
            input_mask=emb_mask,
            n_step=n_step,
            K=K,
            max_output_length=max_output_length,
            mask_id=mask_id,            
            return_intermediates=return_intermediates,
            is_uniform=is_uniform,
            step_size=step_size,
            eps=eps,
            device=device,            
        )

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
    max_output_length: int,
    mask_id: int,
    return_intermediates: Optional[bool] = True,
    is_uniform: Optional[bool] = False,
    step_size: Optional[float] = 0.01,
    eps: Optional[float] = 1e-4,
    device: Optional[torch.DeviceObjType] = "cuda",    
):
    """
    Docstring for sampling_batch
    
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
    """
    model.eval()    

    sp = test_dl.dataset.sp
    hyp = []
    target = []
    
    batch = next(iter(test_dl))
    (
        audio_feats, audio_feat_mask,
        text_feats, text_feat_mask,
        x_1, x_1_mask 
    ) = batch

    audio_feats = audio_feats.to(device)
    audio_feat_mask = audio_feat_mask.to(device)
    # text feature isn't applied yet
    text_feats = text_feats.to(device)
    text_feat_mask = text_feat_mask.to(device)
    x_1 = x_1.to(device)

    emb_seq = audio_feats
    emb_mask = audio_feat_mask
    
    x_1_hat = sampling(
        model=model,            
        input=emb_seq,
        input_mask=emb_mask,
        n_step=n_step,
        K=K,
        max_output_length=max_output_length,
        mask_id=mask_id,            
        return_intermediates=return_intermediates,
        is_uniform=is_uniform,
        step_size=step_size,
        eps=eps,
        device=device,            
    )

    # S, B, T -> B, S, T
    x_1_hat = x_1_hat.permute(1, 0, 2)
    
    hyp = []
    target = []
    for b in range(x_1_hat.size(0)):
        for s in range(x_1_hat.size(1)):
            ids = x_1_hat[b, s].tolist()
            sentence = sp.decode(ids)
            sentence_ids = sp.id_to_piece(ids)
            #print(f"* Step {s}")
            #print(f"  Sentence: {sentence}")
            #print(f"  Tokens: {sentence_ids}")
            
        target_ids = x_1[b].tolist()
        target_sentence = sp.decode(target_ids)
        target_sentence_ids = sp.id_to_piece(target_ids)
        #print(f"TARGET: {target_sentence}")
        #print(f"TARGET ID: {', '.join(target_sentence_ids)}")       
        #print(f"Length: {max_output_length}")     
        target.append(target_sentence)
        hyp.append(sentence)

    return hyp, target
