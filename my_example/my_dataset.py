import os, glob, random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sentencepiece as spm


# my implementation
from make_bpe_model import make_bpe_model

class HuBERTandDeBERTaDataset(Dataset):
    def __init__(self,
                 task="eval",
                 feat_dir="./hubert_deberta_cache",                 
                 bpe_file="./bpe_model/bpe_650.model",
                 K=650):
        self.task = task

        # Resolution K subwords
        self.K = K
        self.file_paths = glob.glob(os.path.join(feat_dir, f"{task}*", "**", "*.pt"))
        self.metadatas = []
        self.sp = spm.SentencePieceProcessor()
        max_audio_length = 0
        max_text_length = 0
        for file in tqdm(self.file_paths):
            data = torch.load(file, map_location="cpu")
            path = file
            audio_length = data["feat_mask"].sum().item()
            text_length = data["text_feat_mask"].sum().item()
            if max_audio_length < audio_length: 
                max_audio_length = audio_length
            if max_text_length < text_length:
                max_text_length = text_length
            metadata = {"path": path,
                        "audio_length": audio_length,
                        "text_length": text_length}
        
            self.metadatas.append(metadata)  

        # sort by length
        self.metadatas.sort(key=lambda x: x["audio_length"])
        
        # loading sentence piece model
        self.sp.Load(bpe_file)

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
        text_feat_mask = data["text_feat_mask"].long()
        gts = data["ground_truth"]
        hyps = data["hypothesis"]
        slus = data["decoupled_normalized_seqlogical"]
        # text -> tensor
        slus = torch.tensor(self.sp.encode(slus)).long()
        slu_mask = torch.ones_like(slus).long()
        return (
            feats, feat_mask,
            text_feats, text_feat_mask, 
            gts, hyps, 
            slus, slu_mask
        )


class MyBatchSampler(Sampler):
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


def my_collate_fn(batch):
    # batch: list of (img, feats(T,H), mask(T), y)    
    (
        feats,              # audio feature
        feat_mask,          # audio feature mask
        text_feats,         # text feature
        text_feat_mask,     # text feature mask
        gts,                # ground truth
        hyps,               # hypothesis
        slus,                # SLU 
        slu_mask           # SLU mask
    ) = zip(*batch)

    feats_padded = pad_sequence(feats, batch_first=True)  # (B, A, D)
    feat_mask_padded = pad_sequence(feat_mask, batch_first=True)  # (B, A)
    text_feats_padded = pad_sequence(text_feats, batch_first=True) # (B, T, D)
    text_feat_mask_padded = pad_sequence(text_feat_mask, batch_first=True) # (B, T)
    slus_padded = pad_sequence(slus, batch_first=True) # (B, T')
    slu_mask_padded = pad_sequence(slu_mask, batch_first=True) # (B, T')

    return (
        feats_padded, feat_mask_padded,
        text_feats_padded, text_feat_mask_padded,
        slus_padded, slu_mask_padded
    )


if __name__ == "__main__":        
    sp = spm.SentencePieceProcessor()
    sp.Load("./bpe_model/bpe_650.model")
    sentence = "[in:send_message [sl:content_exact anyone is going tonight ] ]"
    aa = sp.encode(sentence)
    print(f"{aa=}")
    print(f"{sp.decode(aa)=}")
    batch_size = 2
    dataset = HuBERTandDeBERTaDataset(task="eval", bpe_file="./bpe_model/bpe_650.model")
    data_sampler = MyBatchSampler(dataset, batch_size=batch_size, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_sampler=data_sampler,
        num_workers=0,
        collate_fn=my_collate_fn,        
    )
    
    print(f"{next(iter(dataloader))=}")
