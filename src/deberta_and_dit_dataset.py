import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from my_dataset import Seq2SeqCollator, Seq2SeqDataset
from torch.utils.data import Dataset, Sampler
import sentencepiece as spm

class DeBERTaAndDiTDataset(Dataset):
    def __init__(
        self,
        input_files: str,
        target_files: str,
        in_tokenizer: AutoTokenizer,
        out_tokenizer: spm.SentencePieceProcessor,
        max_len: int = 128,
        add_bos: bool = False,
        add_eos: bool = True,
    ):
        self.input_files = input_files
        self.target_files = target_files
        self.in_tokenizer = in_tokenizer
        self.out_tokenizer = out_tokenizer
        self.max_len = max_len
        # TODO: future work
        self.add_bos = add_bos
        self.add_eos = add_eos

        assert self.in_tokenizer is not None, "Input tokenizer must be provided."
        assert self.out_tokenizer is not None, "Output tokenizer must be provided."

        self.pad_id = self.out_tokenizer.pad_id() if self.out_tokenizer.pad_id() != -1 else 0

        self.input_ids = []
        self.target_ids = []
        self.input_texts = []
        self.target_texts = []
        self.inputs = []

        with open(self.input_files, encoding="utf-8") as fin, \
            open(self.target_files, encoding="utf-8") as ftgt:

            for src_line, tgt_line in zip(fin, ftgt):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue

                self.input_texts.append(src_line)
                self.target_texts.append(tgt_line)
                
                encoded_ids, encoded_all = self.encode_input(src_line)                
                self.input_ids.append(encoded_ids)
                self.inputs.append(encoded_all)
                self.target_ids.append(self.encode_target(tgt_line))

        assert len(self.input_ids) == len(self.target_ids), "Input and target size mismatch."
        assert len(self.input_texts) == len(self.target_texts), "Input and target text size mismatch."
        assert len(self.input_ids) == len(self.inputs), "Input ids and inputs size mismatch."
    
    def encode_input(self, text: str):
        encoded = self.in_tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_tensors="pt"
        )        
        return encoded['input_ids'].squeeze(0), encoded
    
    def encode_target(self, text: str):
        encoded = self.out_tokenizer.encode(text, out_type=int)
        if self.add_bos and self.out_tokenizer.bos_id() != -1:
            encoded = [self.out_tokenizer.bos_id()] + encoded
        if self.add_eos and self.out_tokenizer.eos_id() != -1:
            encoded = encoded + [self.out_tokenizer.eos_id()]
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],    # Shape: (L_src,)
            'target_ids': self.target_ids[idx],  # Shape: (L_tgt,)
            'input_texts': self.input_texts[idx],
            'target_texts': self.target_texts[idx],
            #'inputs': self.inputs[idx],          # Tokenizer output dict
        }


class DeBERTaAndDiTCollator:
    def __init__(self,
                 pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        # batch: List of dicts with 'input' and 'target' tensors
        input_seqs = []
        target_seqs = []
        input_lens = []
        target_lens = []

        for item in batch:
            i_seq = item['input_ids']
            t_seq = item['target_ids']

            input_seqs.append(i_seq)
            target_seqs.append(t_seq)

            input_lens.append(i_seq.size(0))
            target_lens.append(t_seq.size(0))
        
        # Pad sequences
        input_padded = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=self.pad_id)
        target_padded = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=self.pad_id)

        # Create attention masks
        #input_mask = (input_padded != self.pad_id).long()
        #target_mask = (target_padded != self.pad_id).long()
                
        max_input_len = input_padded.shape[1]
        max_target_len = target_padded.shape[1]
        
        input_lens_tensor = torch.tensor(input_lens)
        target_lens_tensor = torch.tensor(target_lens)

        # Broadcasting을 이용한 고속 마스크 생성
        # [0, 1, 2, ...] < [[5], [3], ...] 형태로 비교
        input_mask = (torch.arange(max_input_len) < input_lens_tensor.unsqueeze(1)).long()
        target_mask = (torch.arange(max_target_len) < target_lens_tensor.unsqueeze(1)).long()

        return {
            'input_ids': input_padded,
            'input_masks': input_mask,
            'target_ids': target_padded,
            'target_masks': target_mask
        }
    
class LengthGroupedSampler(Sampler):
    def __init__(self, dataset: DeBERTaAndDiTDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 길이에 따른 인덱스 그룹화
        self.length_buckets = {}
        for idx, target_ids in enumerate(self.dataset.target_ids):
            length = target_ids.size(0)
            if length not in self.length_buckets:
                self.length_buckets[length] = []
            self.length_buckets[length].append(idx)

        # 각 길이 그룹에서 배치 생성
        self.batches = []
        for length, indices in self.length_buckets.items():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                if len(batch_indices) == batch_size or not drop_last:
                    self.batches.append(batch_indices)