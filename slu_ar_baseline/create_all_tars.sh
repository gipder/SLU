#!/bin/bash
# create_all_tars.sh - TAR 파일 한번에 생성 (캐시 포함)

set -e

EXTRACT_FEAT_DIR="../data/slu/hubert_deberta_cache"
TAR_OUTPUT_DIR="../data/slu/hubert_deberta_cache_tar"
TOKENIZER="facebook/hubert-large-ls960-ft"

echo "🚀 TAR 생성 시작 (메타데이터 캐시 자동 생성)"
echo "==========================================="

# Train 데이터
echo ""
echo "📦 Train 데이터 처리..."
python create_tar_dataset.py \
  --source_dir $EXTRACT_FEAT_DIR \
  --output_dir $TAR_OUTPUT_DIR \
  --task train \
  --num_shards 256 \
  --tokenizer_model_name $TOKENIZER

# Eval 데이터들
for eval_task in eval_0 eval_1; do
  if [ -d "$EXTRACT_FEAT_DIR/${eval_task}"* ]; then
    echo ""
    echo "📦 $eval_task 데이터 처리..."
    python create_tar_dataset.py \
      --source_dir $EXTRACT_FEAT_DIR \
      --output_dir $TAR_OUTPUT_DIR \
      --task $eval_task \
      --num_shards 32 \
      --tokenizer_model_name $TOKENIZER
  fi
done

# Test 데이터들
for test_task in test_0 test_1; do
  if [ -d "$EXTRACT_FEAT_DIR/${test_task}"* ]; then
    echo ""
    echo "📦 $test_task 데이터 처리..."
    python create_tar_dataset.py \
      --source_dir $EXTRACT_FEAT_DIR \
      --output_dir $TAR_OUTPUT_DIR \
      --task $test_task \
      --num_shards 64 \
      --tokenizer_model_name $TOKENIZER
  fi
done

echo ""
echo "==========================================="
echo "✅ 모든 TAR 파일 및 캐시 생성 완료!"
echo ""
echo "생성된 파일:"
ls -lh $TAR_OUTPUT_DIR/*.tar $TAR_OUTPUT_DIR/.metadata_cache_*.pkl 2>/dev/null | tail -20
