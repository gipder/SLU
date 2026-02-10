import os, glob, random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sentencepiece as spm
from transformers import AutoProcessor
import tarfile
import io
import pickle
import logging

# Setup module-level logger
logger = logging.getLogger(__name__)

# STOP dataset
# HubertForCTC 모델 기준으로 출력
# Tokenizer: A-Z + blank
# eval: 34 ~ 35
# train: 41 ~ 43
# test: 116 ~ 122 

class HuBERTandDeBERTaDataset(Dataset):
    def __init__(self,
                 task="eval",
                 feat_dir="./hubert_deberta_cache_retrial",
                 tokenizer=None,
                 #max_output_length=128,
                 debugging=False,
                 debugging_num=128,
                 use_tar=False):
        self.task = task
        self.use_tar = use_tar
        self.feat_dir = feat_dir
        self.tokenizer = tokenizer
        
        # tar 파일 핸들을 저장할 딕셔너리
        self.tar_files = {}
        
        if use_tar:
            # tar 파일 모드: tar 파일들에서 인덱스 로드 (메타데이터 자동 캐싱)
            self._load_from_tar(feat_dir, task, debugging, debugging_num)
        else:
            # 기존 방식: 개별 *.pt 파일 로드
            self._load_from_files(feat_dir, task, debugging, debugging_num)

    def _load_from_files(self, feat_dir, task, debugging, debugging_num):
        """기존 방식: 개별 파일들에서 로드"""
        self.file_paths = glob.glob(os.path.join(feat_dir, f"{task}*", "**", "*.pt"), recursive=True)
        self.metadatas = []        
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
        for file in tqdm(self.file_paths, desc="Loading metadata from files"):
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
                        "gt_length": gt_length,
                        "tar_file": None,
                        "tar_member": None}

            self.metadatas.append(metadata)
            idx += 1
            if self.debugging and idx >= self.debugging_num:
                break

        logger.info(f"{self.max_audio_length=}")
        logger.info(f"{self.max_text_feat_length=}")
        logger.info(f"{self.max_text_length=}")
        logger.info(f"{self.max_gt_length=}")
        
        # sort by length
        self.metadatas.sort(key=lambda x: x["audio_length"])

    def _load_from_tar(self, tar_dir, task, debugging, debugging_num):
        """tar 파일들에서 로드 (메타데이터 자동 캐싱)"""
        # 메타데이터 캐시 파일 경로
        cache_file = os.path.join(tar_dir, f".metadata_cache_{task}.pkl")
        
        # 캐시 파일이 있으면 로드
        if os.path.exists(cache_file):
            logger.info(f"Loading metadata from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.metadatas = cache_data['metadatas']
                    self.max_audio_length = cache_data['max_audio_length']
                    self.max_text_feat_length = cache_data['max_text_feat_length']
                    self.max_text_length = cache_data['max_text_length']
                    self.max_gt_length = cache_data['max_gt_length']
                logger.info(f"Loaded {len(self.metadatas)} samples from cache")
                logger.info(f"{self.max_audio_length=}")
                logger.info(f"{self.max_text_feat_length=}")
                logger.info(f"{self.max_text_length=}")
                logger.info(f"{self.max_gt_length=}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, regenerating...")
        
        # tar 파일들 찾기 (.tar.gz 파일도 포함)
        tar_pattern = os.path.join(tar_dir, f"{task}*.tar*")
        tar_files = glob.glob(tar_pattern)
        
        if not tar_files:
            raise ValueError(f"No tar files found matching pattern: {tar_pattern}")
        
        logger.info(f"Found {len(tar_files)} tar files, loading metadata (this may take a while)...")
        
        self.metadatas = []
        self.max_audio_length = 0
        self.max_text_feat_length = 0
        self.max_text_length = 0
        self.max_gt_length = 0
        
        idx = 0
        for tar_path in tqdm(tar_files, desc="Loading metadata from tar files"):
            # tar 파일 열기 (.tar.gz도 자동 지원)
            try:
                tar = tarfile.open(tar_path, 'r:*')
            except Exception as e:
                logger.error(f"Failed to open tar file {tar_path}: {e}")
                continue
                
            self.tar_files[tar_path] = tar
            
            # tar 파일 내의 모든 멤버 순회
            for member in tar.getmembers():
                if member.name.endswith('.pt'):
                    # 메타데이터만 로드 (실제 데이터는 나중에 로드)
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            logger.warning(f"Member {member.name} in {tar_path} is not a regular file, skipping...")
                            continue
                        
                        file_data = f.read()
                        if not file_data:
                            logger.warning(f"Empty data for member {member.name} in {tar_path}, skipping...")
                            continue
                            
                        data = torch.load(io.BytesIO(file_data), map_location="cpu", weights_only=False)
                    except Exception as e:
                        logger.warning(f"Error loading metadata for {member.name} in {tar_path}: {e}, skipping...")
                        continue
                    
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
                    
                    metadata = {
                        "path": member.name,
                        "audio_length": audio_length,
                        "text_feat_length": text_feat_length,
                        "text_length": text_length,
                        "gt_length": gt_length,
                        "tar_file": tar_path,
                        "tar_member": member.name
                    }
                    
                    self.metadatas.append(metadata)
                    idx += 1
                    
                    if debugging and idx >= debugging_num:
                        break
            
            if debugging and idx >= debugging_num:
                break
        
        logger.info(f"{self.max_audio_length=}")
        logger.info(f"{self.max_text_feat_length=}")
        logger.info(f"{self.max_text_length=}")
        logger.info(f"{self.max_gt_length=}")
        
        # sort by length
        self.metadatas.sort(key=lambda x: x["audio_length"])
        
        # 메타데이터 캐싱 (항상 저장)
        logger.info(f"Saving metadata cache to {cache_file}...")
        try:
            cache_data = {
                'metadatas': self.metadatas,
                'max_audio_length': self.max_audio_length,
                'max_text_feat_length': self.max_text_feat_length,
                'max_text_length': self.max_text_length,
                'max_gt_length': self.max_gt_length
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Metadata cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def __len__(self):
        return len(self.metadatas)

    def __getitem__(self, idx):
        item = self.metadatas[idx]
        
        if self.use_tar:
            # tar 파일에서 읽기
            tar_path = item["tar_file"]
            member_name = item["tar_member"]
            
            # tar 파일이 열려있지 않으면 열기
            if tar_path not in self.tar_files:
                self.tar_files[tar_path] = tarfile.open(tar_path, 'r:*')
            
            tar = self.tar_files[tar_path]
            try:
                member = tar.getmember(member_name)
                f = tar.extractfile(member)
                if f is None:
                    raise RuntimeError(f"Failed to extract member: {member_name} is not a regular file")
                file_data = f.read()
                if not file_data:
                    raise RuntimeError(f"Empty data read from: {member_name}")
                data = torch.load(io.BytesIO(file_data), map_location="cpu", weights_only=False)
            except (RuntimeError, KeyError, EOFError, Exception) as e:
                # 실패하면 재시도 (tar 재오픈)
                logger.warning(f"Error loading {member_name}: {str(e)}, retrying with fresh tar file...")
                try:
                    self.tar_files[tar_path].close()
                except:
                    pass
                del self.tar_files[tar_path]
                self.tar_files[tar_path] = tarfile.open(tar_path, 'r:*')
                
                tar = self.tar_files[tar_path]
                member = tar.getmember(member_name)
                f = tar.extractfile(member)
                if f is None:
                    raise RuntimeError(f"Failed to extract member after retry: {member_name} is not a regular file")
                file_data = f.read()
                if not file_data:
                    raise RuntimeError(f"Empty data read after retry from: {member_name}")
                data = torch.load(io.BytesIO(file_data), map_location="cpu", weights_only=False)
        else:
            # 개별 파일에서 읽기
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
    
    def __del__(self):
        """소멸자: 열린 tar 파일들 닫기"""
        if hasattr(self, 'tar_files'):
            for tar in self.tar_files.values():
                try:
                    tar.close()
                except:
                    pass


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
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # tokenizer from HubertForCTC
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

    # 숫자 추가 지금은 안해도 될 듯??
    new_tokens = [str(i) for i in range(10)]
    num_added = processor.tokenizer.add_tokens(new_tokens)
    tokenizer = processor.tokenizer

    batch_size = 4
    
    # 개별 파일 방식 테스트
    logger.info("\n=== Testing with individual files ===")
    mode = "tar" # or "file"
    if mode == "file":
        dataset = HuBERTandDeBERTaDataset(task="train",
                                          feat_dir="./hubert_deberta_cache_retrial",
                                          tokenizer=tokenizer,                                          
                                          use_tar=False,
                                          debugging=True,
                                          debugging_num=16)
    else:
        # tar 파일 방식 테스트
        logger.info("\n=== Testing with tar files ===")
        dataset = HuBERTandDeBERTaDataset(task="eval",
                                          feat_dir="./hubert_deberta_tar",
                                          tokenizer=tokenizer,                                          
                                          use_tar=True,
                                          debugging=False,
                                          debugging_num=16)
    
    data_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_sampler=data_sampler,
        num_workers=0,
        collate_fn=hubert_and_deberta_dataset_collate_fn,
    )

    logger.info(f"\n=== First batch ===")
    batch = next(iter(dataloader))
    feats, feat_mask, text_feats, text_mask, gts, hyps, gt_mask, hyp_mask, gt_strs, hyp_strs = batch
    logger.info(f"feats shape: {feats.shape}")
    logger.info(f"feat_mask shape: {feat_mask.shape}")
    logger.info(f"text_feats shape: {text_feats.shape}")
    logger.info(f"text_mask shape: {text_mask.shape}")
    logger.info(f"gts shape: {gts.shape}")
    logger.info(f"hyps shape: {hyps.shape}")
    logger.info(f"gt_strs: {gt_strs}")
    logger.info(f"hyp_strs: {hyp_strs}")
