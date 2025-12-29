import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel
from dataclasses import dataclass, field

@dataclass
class SLUConfig:
    llm_name: str = "microsoft/deberta-v3-base"
    upsampler_factor: int = 8
    d_model: int = 768
    dropout: float = 0.1
    use_log_probs: bool = True
    blank_token_id: int = 0
    
class SLUModel(nn.Module):
    def __init__(self,
                 config: SLUConfig,
                 **components,                 
                 #upsampler: Optional[nn.Module]=None,
                 #decoder: Optional[nn.Module]=None,
                 #device: Optional[torch.device]=None
    ):
        """
        TODO: REWRITE
        SLUModel that integrates a pre-trained LLM with upsampling and decoding layers.
            llm_name: Name of the pre-trained LLM model.
            upsampler: Upsampling layer/module.
            decoder: Decoding layer/module. In the case of CTC, it is not actually a decoder but a projection layer.
        """
        super().__init__()
        self.config = config        
        self.text_model_name = self.config.llm_name
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name)
        self.components = nn.ModuleDict(components)
        #self.upsampler = upsampler
        #self.decoder = decoder
        self._freeze_text_encoder()
        if "upsampler" in self.components:
            self.upsampler = self.components["upsampler"]
        if "decoder" in self.components:
            self.decoder = self.components["decoder"]
        if "proj" in self.components:
            self.proj = self.components["proj"]
        self.loss = nn.CTCLoss(blank=self.config.blank_token_id,
                               zero_infinity=True) 

    def _freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        print(f"[{self.text_model_name}] parameters are frozen.")

    def forward(self, input_ids, input_masks, target_ids, target_masks):
        # Encode input text
        device = self.device if self.device is not None else input_ids.device
        with torch.inference_mode():
            text_embeddings = self.text_encoder(
                input_ids=input_ids,
                attention_mask=input_masks
            ).last_hidden_state.to(device)

        if "upsampler" in self.components:
            upsampled_embeddings, upsampled_masks = self.upsampler(
                text_embeddings,
                input_masks)
            
        text_embeddings = upsampled_embeddings
        input_masks = upsampled_masks

        B, T, D = text_embeddings.size()        
        input_lengths = input_masks.sum(dim=1).long()
        target_lengths = target_masks.sum(dim=1).long()

        ret = {}
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = self.proj(text_embeddings)            
            if self.config.use_log_probs:
                log_probs = logits.log_softmax(dim=-1)
        
            log_probs = log_probs.transpose(0, 1)  # T x B x Vocab_size
            loss = self.loss(log_probs, target_ids, input_lengths, target_lengths)
            ret['loss'] = loss

        return ret
       

    def save_checkpoint(self, path: str):
        
        torch.save(self.state_dict(), path)
        print(f"Model checkpoint saved at {path}")


    def forward(self, input_ids, attention_mask):
        # Base model forward
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # B x T_src x D

        # Upsample
        upsampled_states = self.upsample(hidden_states)  # B x (T_src * upsample_factor) x D

        # Project to vocab size
        logits = self.proj(upsampled_states)  # B x T_tgt x Vocab_size

        return logits