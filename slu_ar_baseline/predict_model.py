import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

from dit import DiscreteDualDiT
from basic_transformer import BasicTransformer
from intent_predictor import MaskedIntentPredictionModule


@dataclass
class SelfPromptARModelConfig:
    # DIT 설정
    vocab_size: int = 42
    hidden_size: int = 512
    depth: int = 6
    num_heads: int = 8
    audio_dim: int = 1024
    text_dim: int = 1024
    num_intent: int = 80
    sos_token_id: int = 1
    eos_token_id: int = 2
    model_type: str = "transformer"  # "dit" or "transformer"
    norm_first: bool = True  # Whether to apply layer normalization before attention and FFN
    max_output_length: int = 256
    embed_dim: Optional[int] = None
    length_hidden_dim: int = 512
    length_dropout: float = 0.2
    prompt_tag_path: str = "../data/slu/INTENT"
    top_k: int = 1


class SelfPromptARModel(nn.Module):
    def __init__(self, cfg: SelfPromptARModelConfig):
        super().__init__()
        self.cfg = cfg
        self.prompt_table_path: Optional[str] = None
        self.prompt_table_loaded_from_file: bool = False
        if self.cfg.embed_dim is None:
            self.cfg.embed_dim = self.cfg.hidden_size

        self.dit = None
        self.basic_transformer = None
        self.dfm_model = None
        if cfg.model_type == "dit":
            # Not implemented yet, but we can easily add DIT as an alternative to the basic transformer
            self.dit = DiscreteDualDiT(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                audio_dim=cfg.audio_dim,
                text_dim=cfg.text_dim,
            )
            self.slu_model = self.dit
        elif cfg.model_type == "transformer":
            self.basic_transformer = BasicTransformer(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                audio_dim=cfg.audio_dim,
                text_dim=cfg.text_dim,
                max_output_length=cfg.max_output_length,
                norm_first=cfg.norm_first,
            )
            self.slu_model = self.basic_transformer
        else:
            raise ValueError(f"Unknown model_type: {cfg.model_type}")
        
        self.prompt_predictor = MaskedIntentPredictionModule(
            embed_dim=cfg.embed_dim,
            length_hidden_dim=cfg.length_hidden_dim,
            num_intent=cfg.num_intent,
            length_dropout=cfg.length_dropout
        )

        self.prompt_table = self._load_prompt_table(cfg.prompt_tag_path)

    def _resolve_prompt_path(self, prompt_tag_path: str) -> Path:
        path = Path(prompt_tag_path)
        if path.is_absolute():
            return path
        # Resolve relative path robustly against common execution roots.
        # Priority: current working dir -> this file dir -> workspace root.
        module_dir = Path(__file__).resolve().parent
        workspace_root = module_dir.parent
        candidates = [
            Path.cwd() / path,
            module_dir / path,
            workspace_root / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Return cwd-based absolute path for clearer downstream diagnostics.
        return (Path.cwd() / path).resolve()

    def _load_prompt_table(self, prompt_tag_path: str):
        path = self._resolve_prompt_path(prompt_tag_path)
        self.prompt_table_path = str(path)
        if not path.exists():
            self.prompt_table_loaded_from_file = False
            warnings.warn(
                f"prompt_tag_path not found: {path}. Falling back to synthetic INTENT labels.",
                RuntimeWarning,
            )
            return [f"INTENT_{i}" for i in range(self.cfg.num_intent)]

        self.prompt_table_loaded_from_file = True
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) < self.cfg.num_intent:
            lines.extend([f"INTENT_{i}" for i in range(len(lines), self.cfg.num_intent)])
        return lines[: self.cfg.num_intent]

    @staticmethod
    def _to_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # input mask convention: 1/True=valid, 0/False=pad
        # padding mask convention: True=pad
        if mask is None:
            return None
        return ~mask.bool()

    def _build_intent_inputs(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ):
        if self.basic_transformer is None:
            raise NotImplementedError("intent retrieval is currently supported with model_type='transformer'.")

        memory_list = []
        mask_list = []

        audio_padding_mask = self._to_padding_mask(audio_mask)
        text_padding_mask = self._to_padding_mask(text_mask)

        if audio_feats is not None:
            audio = self.basic_transformer.audio_proj(audio_feats) + self.basic_transformer.modality_emb[0]
            memory_list.append(audio)
            if audio_padding_mask is not None:
                mask_list.append(audio_padding_mask)

        if text_feats is not None:
            text = self.basic_transformer.text_proj(text_feats) + self.basic_transformer.modality_emb[1]
            memory_list.append(text)
            if text_padding_mask is not None:
                mask_list.append(text_padding_mask)

        if not memory_list:
            raise ValueError("At least one of audio_feats or text_feats must be provided.")

        if len(memory_list) == 1:
            memory = memory_list[0]
            memory_padding_mask = mask_list[0] if mask_list else None
        else:
            memory = torch.cat(memory_list, dim=1)
            memory_padding_mask = torch.cat(mask_list, dim=1) if mask_list else None

        return memory, memory_padding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        text_feats: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        audio_padding_mask = self._to_padding_mask(audio_mask)
        text_padding_mask = self._to_padding_mask(text_mask)
        
        logits = self.slu_model(
            input_ids,
            audio_feats, text_feats,
            audio_padding_mask, text_padding_mask
        )

        intent_x, intent_padding_mask = self._build_intent_inputs(
            audio_feats=audio_feats,
            text_feats=text_feats,
            audio_mask=audio_mask,
            text_mask=text_mask,
        )
        prompt_logits = self.prompt_predictor(intent_x, intent_padding_mask)

        return logits, prompt_logits

    @torch.no_grad()
    def retrieve_prompts(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        top_k: int = 1,
    ):
        intent_x, intent_padding_mask = self._build_intent_inputs(
            audio_feats=audio_feats,
            text_feats=text_feats,
            audio_mask=audio_mask,
            text_mask=text_mask,
        )
        prompt_logits = self.prompt_predictor(intent_x, intent_padding_mask)
        prompt_probs = torch.softmax(prompt_logits, dim=-1)

        k = max(1, min(int(top_k), self.cfg.num_intent))
        top_scores, top_ids = torch.topk(prompt_probs, k=k, dim=-1)
        #print(f"{top_ids=}, {top_scores=}")

        retrieved_prompts = []
        for row in top_ids.tolist():
            retrieved_prompts.append([self.prompt_table[idx] for idx in row])

        return {
            "prompt_logits": prompt_logits,
            "prompt_probs": prompt_probs,
            "top_intent_ids": top_ids,
            "top_intent_scores": top_scores,
            "retrieved_prompts": retrieved_prompts,
        }

    @torch.no_grad()
    def decode(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        max_output_length: Optional[int] = None,
        sos_id: int = 1,
        eos_id: Optional[int] = 2,
        use_cache: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        audio_padding_mask = self._to_padding_mask(audio_mask)
        text_padding_mask = self._to_padding_mask(text_mask)

        return self.slu_model.decode(
            audio_feats, text_feats, audio_padding_mask, text_padding_mask,
            max_output_length, sos_id, eos_id, use_cache=use_cache, device=device
        )

    @torch.no_grad()
    def decode_with_prompt(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        prompt_prefix_ids: Optional[torch.Tensor] = None,  # (B, P) pre-tokenized intent prefix
        max_output_length: Optional[int] = None,
        sos_id: int = 1,
        eos_id: int = 2,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Prompt-conditioned autoregressive decoding.

        Flow:
          1. Build audio/text conditioning memory once (same projection as BasicTransformer).
          2. Pre-fill decoder with [SOS, *prompt_prefix_ids]  ← teacher-forced intent prefix.
          3. Continue free autoregressive generation until EOS or max_output_length.

        Args:
            prompt_prefix_ids: (B, P) int64 tensor of already-tokenized intent prefix tokens.
                               Obtain by tokenizing retrieve_prompts()['retrieved_prompts'].
                               If None, behaves identically to slu_model.decode().

        Returns:
            generated: (B, T) token ids  (includes SOS + prefix + generated tail)
        """
        bt = self.basic_transformer
        if bt is None:
            raise NotImplementedError("decode_with_prompt requires model_type='transformer'.")

        if device is None:
            device = audio_feats.device if audio_feats is not None else text_feats.device

        max_output_length = max_output_length or bt.pos_emb.size(1)

        audio_padding_mask = self._to_padding_mask(audio_mask)
        text_padding_mask  = self._to_padding_mask(text_mask)

        # ── Step 1: Pre-compute conditioning memory (once, reused every step) ──
        memory, memory_key_padding_mask = bt._build_memory(
            audio_feats, text_feats, audio_padding_mask, text_padding_mask
        )

        B = memory.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # ── Step 2: Pre-fill generated sequence with [SOS, *prompt_prefix] ────
        sos_t = torch.full((B, 1), fill_value=sos_id, dtype=torch.long, device=device)
        if prompt_prefix_ids is not None:
            # prompt_prefix_ids: (B, P)  — already tokenized intent prefix
            generated = torch.cat([sos_t, prompt_prefix_ids.to(device)], dim=1)
            # Mark rows that already contain EOS inside the prefix
            finished |= (generated == eos_id).any(dim=1)
        else:
            generated = sos_t

        # ── Step 3: Autoregressive generation from the end of the prefix ──────
        while generated.size(1) < max_output_length:
            seq_len = generated.size(1)
            if seq_len > bt.pos_emb.size(1):
                break

            x = bt.token_emb(generated) + bt.pos_emb[:, :seq_len, :]
            tgt_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
            )
            x = bt.decoder(
                x, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            logits = bt.head(x[:, -1, :])            # only last position
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Keep already-finished rows frozen at EOS
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, eos_id),
                next_token,
            )

            generated = torch.cat([generated, next_token], dim=1)
            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return generated

    @torch.no_grad()
    def decode_with_retrieval(
        self,
        audio_feats: torch.Tensor,
        text_feats: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        tokenizer=None,
        top_k_retrieval: int = 1,
        max_output_length: Optional[int] = None,
        sos_id: int = 1,
        eos_id: int = 2,
        device: Optional[torch.device] = None,
    ):
        """
        Full pipeline: intent retrieval → tokenize prompt → prompt-conditioned decode.

        Args:
            tokenizer: object with .encode(str) -> List[int] interface.
                       If None, decoding proceeds without any prompt prefix.
            top_k_retrieval: how many top intents to retrieve (uses rank-1 as prefix).

        Returns:
            dict with keys:
              - generated:          (B, T) decoded token ids
              - retrieved_prompts:  list[list[str]] top-k intent labels per batch
              - prompt_prefix_ids:  (B, P) tokenized prefix actually used (or None)
        """
        # ── Retrieve intent prompts ──────────────────────────────────────────
        retrieval = self.retrieve_prompts(
            audio_feats=audio_feats,
            text_feats=text_feats,
            audio_mask=audio_mask,
            text_mask=text_mask,
            top_k=top_k_retrieval,
        )

        # ── Tokenize top-1 intent for each sample in the batch ───────────────
        prompt_prefix_ids = None
        if tokenizer is not None:
            prefix_id_list = []
            for row_labels in retrieval["retrieved_prompts"]:
                top_label = row_labels[0]          # rank-1 intent string
                ids = tokenizer.encode(top_label)  # list[int]
                prefix_id_list.append(ids)

            # Pad to same length within the batch            
            max_p = max(len(ids) for ids in prefix_id_list)
            pad_id = getattr(tokenizer, "pad_token_id", 0)
            padded = [ids + [pad_id] * (max_p - len(ids)) for ids in prefix_id_list]
            prompt_prefix_ids = torch.tensor(padded, dtype=torch.long)            

        # ── Prompt-conditioned decode ─────────────────────────────────────────
        generated = self.decode_with_prompt(
            audio_feats=audio_feats,
            text_feats=text_feats,
            audio_mask=audio_mask,
            text_mask=text_mask,
            prompt_prefix_ids=prompt_prefix_ids,
            max_output_length=max_output_length,
            sos_id=sos_id,
            eos_id=eos_id,
            device=device,
        )

        return {
            "generated": generated,
            "retrieved_prompts": retrieval["retrieved_prompts"],
            "prompt_prefix_ids": prompt_prefix_ids,
            "top_intent_ids": retrieval["top_intent_ids"],
            "top_intent_scores": retrieval["top_intent_scores"],
        }


if __name__ == "__main__":
    import time

    torch.manual_seed(42)

    B = 2
    K = 50
    T_out = 50
    D = 512
    n_H = 8

    cfg = SelfPromptARModelConfig(
        vocab_size=K,
        hidden_size=D,
        audio_dim=D,
        text_dim=D,
        depth=4,
        num_heads=n_H,
        max_output_length=T_out,
        model_type="transformer",
        num_intent=80,
        length_hidden_dim=512,
        length_dropout=0.1,
    )

    print("=" * 60)
    print(f"Config: {cfg}")
    print("=" * 60)

    model = SelfPromptARModel(cfg)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Dummy inputs ─────────────────────────────────────────────────────────
    input_ids   = torch.randint(3, K, (B, T_out))
    audio_feats = torch.rand(B, T_out * 4, D)
    audio_mask  = torch.ones(B, T_out * 4, dtype=torch.bool)
    text_feats  = torch.rand(B, T_out * 2, D)
    text_mask   = torch.ones(B, T_out * 2, dtype=torch.bool)

    # ── Test 1: forward ───────────────────────────────────────────────────────
    print("[Test 1] forward()")
    logits, prompt_logits = model(input_ids, audio_feats, audio_mask, text_feats, text_mask)
    print(f"  logits       : {logits.shape}")        # (B, T_out, K)
    print(f"  prompt_logits: {prompt_logits.shape}")  # (B, num_intent)

    # ── Test 2: retrieve_prompts ──────────────────────────────────────────────
    print("\n[Test 2] retrieve_prompts(top_k=3)")
    retrieval = model.retrieve_prompts(
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
        top_k=3,
    )
    print(f"  top_intent_ids   : {retrieval['top_intent_ids']}")
    print(f"  top_intent_scores: {retrieval['top_intent_scores'].round(decimals=4)}")
    print(f"  retrieved_prompts: {retrieval['retrieved_prompts']}")

    # ── Test 3: baseline decode (no prompt) ──────────────────────────────────
    print("\n[Test 3] decode()  (baseline — no prompt)")
    t0 = time.time()
    decoded_ids = model.decode(
        audio_feats, text_feats, audio_mask, text_mask,
        max_output_length=T_out, sos_id=1, eos_id=2, use_cache=False,
    )
    print(f"  shape   : {decoded_ids.shape}")
    print(f"  tokens  : {decoded_ids[0].tolist()}")
    print(f"  elapsed : {time.time() - t0:.3f}s")

    # ── Test 4: decode_with_prompt (pre-tokenized prefix) ────────────────────
    print("\n[Test 4] decode_with_prompt()  (with tokenized prompt prefix)")
    # Simulate: top-1 intent token IDs, e.g. 2 tokens per sample
    # In practice: tokenizer.encode(retrieval['retrieved_prompts'][b][0])
    P = 2
    prompt_prefix_ids = torch.randint(3, K, (B, P))
    print(f"  simulated prefix ids: {prompt_prefix_ids.tolist()}")

    t0 = time.time()
    decoded_with_prompt = model.decode_with_prompt(
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
        prompt_prefix_ids=prompt_prefix_ids,
        max_output_length=T_out,
        sos_id=1,
        eos_id=2,
    )
    print(f"  shape            : {decoded_with_prompt.shape}")
    print(f"  tokens (batch 0) : {decoded_with_prompt[0].tolist()}")
    print(f"    [0]=SOS({1}), [1..{P}]=prefix, [{P+1}..]=generated")
    print(f"  elapsed : {time.time() - t0:.3f}s")

    # tokenizer
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    # adding numbers from 0 to 9 + "[MASK]" if not already present
    additional_tokens = ["[", "]", ":", "_"]
    new_tokens = [str(i) for i in range(10)] + ["[MASK]"] + additional_tokens
    num_added = processor.tokenizer.add_tokens(new_tokens)    
    tokenizer = processor.tokenizer

    # ── Test 5: decode_with_retrieval (full pipeline, no tokenizer) ──────────
    print("\n[Test 5] decode_with_retrieval()  (full pipeline, tokenizer=None)")
    result = model.decode_with_retrieval(
        audio_feats=audio_feats,
        text_feats=text_feats,
        audio_mask=audio_mask,
        text_mask=text_mask,
        tokenizer=tokenizer,          # no tokenizer → prompt_prefix_ids=None
        top_k_retrieval=1,
        max_output_length=T_out,
        sos_id=1,
        eos_id=2,
    )
    print(f"  generated shape   : {result['generated'].shape}")
    print(f"  retrieved_prompts : {result['retrieved_prompts']}")
    print(f"  prompt_prefix_ids : {result['prompt_prefix_ids']}")  # None when no tokenizer

    # ── Sanity: compare decode vs decode_with_prompt(prefix=None) ────────────
    print("\n[Sanity] decode() vs decode_with_prompt(prefix=None) should match")
    dec_no_prefix = model.decode_with_prompt(
        audio_feats, text_feats, audio_mask, text_mask,
        prompt_prefix_ids=None, max_output_length=T_out, sos_id=1, eos_id=2,
    )
    print(f"  decode()               tokens: {decoded_ids[0].tolist()}")
    print(f"  decode_with_prompt()   tokens: {dec_no_prefix[0].tolist()}")
    match = decoded_ids[0].tolist() == dec_no_prefix[0].tolist()
    print(f"  Outputs match: {match}")
