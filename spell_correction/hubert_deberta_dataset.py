import os, glob, random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sentencepiece as spm
from transformers import AutoProcessor

# STOP dataset
# HubertForCTC 모델 기준으로 출력
# Tokenizer: A-Z + blank
# eval: 34 ~ 35
# train: 41 ~ 43
# test: 116 ~ 122 

class HuBERTandDeBERTaDataset(Dataset):
    def __init__(self,
                 task="eval",
                 feat_dir="./hubert_cache",
                 tokenizer=None,
                 #max_output_length=128,
                 debugging=False,
                 debugging_num=128,):
        self.task = task

        self.file_paths = glob.glob(os.path.join(feat_dir, f"{task}*", "**", "*.pt"))
        self.metadatas = []
        self.tokenizer = tokenizer        
        #self.max_output_length = max_output_length

        # 가장 긴 오디오, 텍스트 길이 추적용
        self.max_audio_length = 0
        self.max_text_feat_length = 0
        self.max_text_length = 0
        self.max_gt_length = 0

        # 디버깅 모드
        self.debugging = debugging
        self.debugging_num = debugging_num

        idx = 0
        for file in tqdm(self.file_paths):
            data = torch.load(file, map_location="cpu")
            path = file
            audio_length = data["feat_mask"].sum().item() 
            text_feat_length = data["text_feat_mask"].sum().item()           
            hypothesis = data["greedy_hypothesis"]
            gt = data["ground_truth"]
            text_length = len(self.tokenizer.encode(hypothesis))
            gt_length = len(self.tokenizer.encode(gt))
            
            if self.max_text_feat_length < text_feat_length:
                self.max_text_feat_length = text_feat_length
            if self.max_audio_length < audio_length:
                self.max_audio_length = audio_length
            if self.max_text_length < text_length:
                self.max_text_length = text_length
            if self.max_gt_length < gt_length:
                self.max_gt_length = gt_length
            metadata = {"path": path,
                        "audio_length": audio_length,
                        "text_feat_length": text_feat_length,
                        "text_length": text_length,
                        "gt_length": gt_length}

            self.metadatas.append(metadata)
            idx += 1
            if self.debugging and idx >= self.debugging_num:
                break

        print(f"{self.max_audio_length=}")
        print(f"{self.max_text_feat_length=}")
        print(f"{self.max_text_length=}")
        print(f"{self.max_gt_length=}")
        
        # sort by length
        self.metadatas.sort(key=lambda x: x["audio_length"])

    def __len__(self):
        return len(self.metadatas)

    def __getitem__(self, idx):
        item = self.metadatas[idx]
        path = item["path"]

        data = torch.load(path, map_location="cpu")
        # audio_feat, audio_feat_mask, text_feat, text_feat_mask, gt, hyp, slu
        feats = data["feats"]
        feat_mask = data["feat_mask"].long()   
        text_feats = data["text_feats"]
        text_mask = data["text_feat_mask"].long()    
        str_gts = data["ground_truth"]
        str_hyps = data["greedy_hypothesis"]
                
        gts_encoded = self.tokenizer.encode(str_gts)
        hyps_encoded = self.tokenizer.encode(str_hyps)        
        gts = torch.tensor(gts_encoded).long()
        hyps = torch.tensor(hyps_encoded).long()
        #make mask
        gt_mask = torch.ones_like(gts).long()
        hyp_mask = torch.ones_like(hyps).long()
                
        return (
            feats, feat_mask,
            text_feats, text_mask,
            gts, hyps,
            gt_mask, hyp_mask,
            str_gts, str_hyps,
        )


class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data_source)))

        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def hubert_and_deberta_dataset_collate_fn(batch):
    # batch: list of (img, feats(T,H), mask(T), y)
    (
        feats,              # audio feature
        feat_mask,          # audio feature mask
        text_feats,         # text feature
        text_mask,          # text feature mask
        gts,                # ground truth
        hyps,               # hypothesis  
        gt_mask,            # ground truth mask
        hyp_mask,           # hypothesis mask
        gt_strs,            # ground truth strings
        hyp_strs,           # hypothesis strings
    ) = zip(*batch)
    
    
    feats_padded = pad_sequence(feats, batch_first=True)  # (B, A, D)
    feat_mask_padded = pad_sequence(feat_mask, batch_first=True)  # (B, A)
    text_feats_padded = pad_sequence(text_feats, batch_first=True)  # (B, T, D)
    text_mask_padded = pad_sequence(text_mask, batch_first=True)  # (B, T)

    gts_padded = pad_sequence(gts, batch_first=True) # (B, T')
    hyps_padded = pad_sequence(hyps, batch_first=True) # (B, T')
    gt_mask_padded = pad_sequence(gt_mask, batch_first=True) # (B, T')
    hyp_mask_padded = pad_sequence(hyp_mask, batch_first=True) # (B

    return (
        feats_padded, feat_mask_padded,
        text_feats_padded, text_mask_padded,
        gts_padded, hyps_padded,  
        gt_mask_padded, hyp_mask_padded,
        gt_strs, hyp_strs,
    )


if __name__ == "__main__":
    # tokenizer from HubertForCTC
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    # 숫자 추가 지금은 안해도 될 듯??
    new_tokens = [str(i) for i in range(10)]
    num_added = processor.tokenizer.add_tokens(new_tokens)
    tokenizer = processor.tokenizer

    batch_size = 4
    dataset = HuBERTandDeBERTaDataset(task="test",
                                      tokenizer=tokenizer,
                                      feat_dir="./hubert_deberta_cache_retrial",
                                      debugging=True,
                                      debugging_num=16)
    data_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_sampler=data_sampler,
        num_workers=0,
        collate_fn=hubert_and_deberta_dataset_collate_fn,
    )

    #print(f"{next(iter(dataloader))=}")
