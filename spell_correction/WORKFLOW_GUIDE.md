# 개선된 워크플로우 가이드

## 이전 워크플로우 (3단계)
1. `extract_feature.py` - 개별 `.pt` 파일 생성
2. `create_tar_dataset.py` - tar 아카이브로 압축
3. `eval.py` or `train.py` - 모델 실행

## 새로운 워크플로우 (2단계)
1. `extract_feature.py` - 개별 `.pt` 파일 생성
2. `create_tar_dataset.py` (개선됨) - tar 아카이브 + 메타데이터 캐시 동시 생성

---

## 상세 사용법

### 1단계: 피처 추출
```bash
cd spell_correction
python extract_feature.py
```

개별 `.pt` 파일들이 생성됩니다:
- `./hubert_deberta_cache_retrial/train/.../*.pt`
- `./hubert_deberta_cache_retrial/eval_0/.../*.pt`
- `./hubert_deberta_cache_retrial/test_0/.../*.pt`

### 2단계: TAR 아카이브 + 캐시 생성 (개선됨)

**Train 데이터:**
```bash
python create_tar_dataset.py \
  --source_dir ./hubert_deberta_cache_retrial \
  --output_dir ./hubert_deberta_tar \
  --task train \
  --num_shards 4 \
  --tokenizer_model_name facebook/hubert-large-ls960-ft
```

**Eval 데이터:**
```bash
python create_tar_dataset.py \
  --source_dir ./hubert_deberta_cache_retrial \
  --output_dir ./hubert_deberta_tar \
  --task eval_0 \
  --num_shards 2 \
  --tokenizer_model_name facebook/hubert-large-ls960-ft
```

**Test 데이터:**
```bash
python create_tar_dataset.py \
  --source_dir ./hubert_deberta_cache_retrial \
  --output_dir ./hubert_deberta_tar \
  --task test_0 \
  --num_shards 2 \
  --tokenizer_model_name facebook/hubert-large-ls960-ft
```

### 결과물

각 task별로 생성되는 파일:
```
./hubert_deberta_tar/
├── train_shard_0000.tar
├── train_shard_0001.tar
├── train_shard_0002.tar
├── train_shard_0003.tar
├── .metadata_cache_train.pkl          # 🆕 캐시 파일
├── eval_0_shard_0000.tar
├── eval_0_shard_0001.tar
├── .metadata_cache_eval_0.pkl         # 🆕 캐시 파일
├── test_0_shard_0000.tar
├── test_0_shard_0001.tar
└── .metadata_cache_test_0.pkl         # 🆕 캐시 파일
```

### 3단계: 모델 실행

캐시 파일이 생성되었으므로, `eval.py` 실행 시 처음 로드가 훨씬 빠릅니다:

```bash
python eval.py \
  --ckpt_path ./baseline_additional_loss/model_step300000.pt \
  --batch_size 256 \
  --num_workers 4 \
  --dataset_path ./hubert_deberta_tar \
  --use_tar True
```

---

## 주요 개선사항

### ✅ 장점
1. **캐시 자동 생성** - tar 생성 시 메타데이터 캐시가 함께 생성됨
2. **첫 로드 시간 단축** - 캐시가 있으면 메타데이터 로드가 거의 즉시 완료
3. **원클릭 처리** - `create_tar_dataset.py`에서 토크나이저 옵션만 추가하면 끝
4. **안정성** - 토크나이저 로드 실패 시에도 tar 생성은 계속됨

### 📊 성능 비교

**첫 실행 시 (캐시 없음):**
- 이전: tar 파일을 읽으면서 메타데이터 추출 (느림)
- 개선: tar 생성 단계에서 이미 캐시 생성 완료 (빠름)

**두 번째 이후 실행 (캐시 있음):**
- 캐시 파일만 로드 (매우 빠름)

---

## 옵션 설명

### `create_tar_dataset.py`
- `--tokenizer_model_name`: 토크나이저 모델 (예: `facebook/hubert-large-ls960-ft`)
  - 지정하면 메타데이터 캐시 자동 생성
  - 미지정하면 tar만 생성 (캐시는 나중에 첫 실행 시 생성됨)

### `eval.py`
- `--use_tar`: `True` - tar 파일 사용 (권장)
- `--dataset_path`: tar 파일 위치

---

## 주의사항

1. **토크나이저 설치**: `transformers` 라이브러리 필요
2. **메모리**: tar 생성 중 메타데이터 추출에 메모리 사용
3. **Shard 개수**: 너무 많으면 tar 파일이 커지고, 너무 적으면 개별 파일이 커짐

---

## 스크립트 한번에 실행 (Bash)

```bash
#!/bin/bash

EXTRACT_FEAT_DIR="./hubert_deberta_cache_retrial"
TAR_OUTPUT_DIR="./hubert_deberta_tar"
TOKENIZER="facebook/hubert-large-ls960-ft"

# TAR 생성 (모든 task)
for task in train eval_0 eval_1 test_0 test_1; do
  python create_tar_dataset.py \
    --source_dir $EXTRACT_FEAT_DIR \
    --output_dir $TAR_OUTPUT_DIR \
    --task $task \
    --num_shards 4 \
    --tokenizer_model_name $TOKENIZER   
done

echo "✅ All tar files and cache files created!"
```
