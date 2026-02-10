"""
*.pt 파일들을 tar 아카이브로 압축하고 여러 개로 분할하는 스크립트
동시에 메타데이터 캐시 파일도 생성

사용법:
    python create_tar_dataset.py --source_dir ./hubert_deberta_cache_retrial \
                                  --output_dir ./hubert_deberta_tar \
                                  --task train \
                                  --num_shards 4
"""

import os
import glob
import tarfile
import argparse
from tqdm import tqdm
import torch
import io
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def create_tar_archives(source_dir, output_dir, task, num_shards=4, tokenizer=None):
    """
    *.pt 파일들을 여러 개의 tar 아카이브로 압축하고 메타데이터 캐시 생성
    
    Args:
        source_dir: 원본 데이터 디렉토리
        output_dir: tar 파일을 저장할 디렉토리
        task: task 이름 (예: train, eval_0, eval_1, test_0, test_1)
        num_shards: 몇 개의 tar 파일로 분할할지
        tokenizer: 토크나이저 (캐시 생성을 위해 필요)
    """
    # 모든 .pt 파일 찾기
    pattern = os.path.join(source_dir, f"{task}*", "**", "*.pt")
    pt_files = glob.glob(pattern, recursive=True)
    
    if not pt_files:
        print(f"No .pt files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(pt_files)} .pt files for task '{task}'")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일들을 shard별로 나누기
    files_per_shard = (len(pt_files) + num_shards - 1) // num_shards
    
    # 메타데이터 수집 (모든 파일)
    all_metadatas = []
    max_audio_length = 0
    max_text_feat_length = 0
    max_text_length = 0
    max_gt_length = 0
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * files_per_shard
        end_idx = min((shard_idx + 1) * files_per_shard, len(pt_files))
        shard_files = pt_files[start_idx:end_idx]
        
        if not shard_files:
            continue
        
        # tar 파일 이름 (압축 없음)
        tar_filename = os.path.join(output_dir, f"{task}_shard_{shard_idx:04d}.tar")
        
        print(f"\nCreating {tar_filename} with {len(shard_files)} files...")
        
        with tarfile.open(tar_filename, 'w:') as tar:
            for file_path in tqdm(shard_files, desc=f"Shard {shard_idx} / {num_shards}"):
                # source_dir를 기준으로 상대 경로 생성
                arcname = os.path.relpath(file_path, source_dir)
                tar.add(file_path, arcname=arcname)
                
                # 메타데이터 추출
                if tokenizer is not None:
                    try:
                        data = torch.load(file_path, map_location="cpu")
                        audio_length = data["feat_mask"].sum().item()
                        text_feat_length = data["text_feat_mask"].sum().item()
                        hypothesis = data["greedy_hypothesis"]
                        gt = data["ground_truth"]
                        text_length = len(tokenizer.encode(hypothesis))
                        gt_length = len(tokenizer.encode(gt))
                        
                        if max_text_feat_length < text_feat_length:
                            max_text_feat_length = text_feat_length
                        if max_audio_length < audio_length:
                            max_audio_length = audio_length
                        if max_text_length < text_length:
                            max_text_length = text_length
                        if max_gt_length < gt_length:
                            max_gt_length = gt_length
                        
                        metadata = {
                            "path": arcname,
                            "audio_length": audio_length,
                            "text_feat_length": text_feat_length,
                            "text_length": text_length,
                            "gt_length": gt_length,
                            "tar_file": tar_filename,
                            "tar_member": arcname
                        }
                        all_metadatas.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata from {file_path}: {e}")
                        continue
        
        print(f"Created {tar_filename} ({os.path.getsize(tar_filename) / 1024 / 1024:.2f} MB)")
    
    # 메타데이터 캐시 저장 (tokenizer가 제공된 경우에만)
    if tokenizer is not None and all_metadatas:
        all_metadatas.sort(key=lambda x: x["audio_length"])
        
        cache_file = os.path.join(output_dir, f".metadata_cache_{task}.pkl")
        logger.info(f"\nSaving metadata cache to {cache_file}...")
        try:
            cache_data = {
                'metadatas': all_metadatas,
                'max_audio_length': max_audio_length,
                'max_text_feat_length': max_text_feat_length,
                'max_text_length': max_text_length,
                'max_gt_length': max_gt_length
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Metadata cache saved successfully: {len(all_metadatas)} samples")
            logger.info(f"  max_audio_length={max_audio_length}")
            logger.info(f"  max_text_feat_length={max_text_feat_length}")
            logger.info(f"  max_text_length={max_text_length}")
            logger.info(f"  max_gt_length={max_gt_length}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


def list_tar_contents(tar_path):
    """tar 파일의 내용물 확인"""
    print(f"\nContents of {tar_path}:")
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        print(f"Total files: {len(members)}")
        print("First 10 files:")
        for member in members[:10]:
            print(f"  {member.name} ({member.size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tar archives from .pt files with metadata cache")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Source directory containing .pt files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for tar files")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., train, eval_0, eval_1, test_0, test_1)")
    parser.add_argument("--num_shards", type=int, default=4,
                        help="Number of tar files to create (default: 4)")
    parser.add_argument("--tokenizer_model_name", type=str, default=None,
                        help="Tokenizer model name to extract metadata (e.g., facebook/hubert-large-ls960-ft)")
    parser.add_argument("--list_contents", action="store_true",
                        help="List contents of first tar file after creation")
    
    args = parser.parse_args()
    
    # 토크나이저 로드 (메타데이터 추출용)
    tokenizer = None
    if args.tokenizer_model_name:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(args.tokenizer_model_name)
            tokenizer = processor.tokenizer
            logger.info(f"Loaded tokenizer from {args.tokenizer_model_name}")
            
            # 숫자 토큰 추가
            new_tokens = [str(i) for i in range(10)]
            num_added = tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {num_added} number tokens to tokenizer")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, cache will not be created")
            tokenizer = None
    else:
        logger.info("Tokenizer not specified, cache file will not be created. Use --tokenizer_model_name to create cache.")
    
    create_tar_archives(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        task=args.task,
        num_shards=args.num_shards,
        tokenizer=tokenizer
    )
    
    # 첫 번째 tar 파일 내용 확인
    if args.list_contents:
        first_tar = os.path.join(args.output_dir, f"{args.task}_shard_0000.tar")
        if os.path.exists(first_tar):
            list_tar_contents(first_tar)
