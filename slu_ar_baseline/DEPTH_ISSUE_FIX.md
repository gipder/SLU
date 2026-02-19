# Depth 6+ 실험 안정성 문제 해결

## 문제 증상
- Depth 6 이상에서 학습 중 갑자기 성능이 붕괴 (WER 100%)
- Epoch 15: 63.87% 정확도 → Epoch 20: 0% 정확도

## 근본 원인
1. **Post-LayerNorm 사용**: PyTorch 기본 TransformerDecoder는 post-norm 구조를 사용하는데, depth가 깊을 때 gradient exploding/vanishing 문제에 취약
2. **초기화 부재**: modality embedding, projection layers, output head의 proper initialization이 없음
3. **Mixed Precision 비활성화**: `autocast(enabled=False)`로 numerical stability 혜택을 받지 못함
4. **Gradient 모니터링 부재**: gradient explosion을 조기에 감지하지 못함

## 적용된 수정사항

### 1. Pre-LayerNorm 구조로 변경 ([basic_transformer.py](basic_transformer.py))
```python
decoder_layer = nn.TransformerDecoderLayer(
    ...
    norm_first=True,  # Pre-LN for stability with deep models
)
```
**효과**: Residual connection 전에 normalization을 적용하여 gradient flow 안정화

### 2. Proper Parameter Initialization
```python
# Positional & modality embeddings
nn.init.normal_(self.pos_emb, std=0.02)
nn.init.normal_(self.modality_emb, std=0.02)

# Projection layers
nn.init.xavier_uniform_(self.audio_proj.weight)
nn.init.xavier_uniform_(self.text_proj.weight)

# Output head with small weights
nn.init.normal_(self.head.weight, std=0.02)
```
**효과**: 초기 activation의 분산을 적절하게 유지하여 안정적인 학습 시작

### 3. Mixed Precision Training 활성화 ([train.py](train.py))
```python
with torch.amp.autocast('cuda', enabled=use_cuda):  # Changed from enabled=False
```
**효과**: FP16/FP32 혼용으로 numerical stability 향상 및 메모리 절약

### 4. Gradient Monitoring 추가
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
logger.info(f"... grad_norm={grad_norm:.4f}, scale={scaler.get_scale():.1f}")
```
**효과**: Gradient explosion을 실시간으로 모니터링하여 조기 감지 가능

## 추가 권장사항

### Depth별 학습률 조정
```bash
# depth 6: lr=1e-4 (현재 사용 중)
# depth 8: lr=5e-5 ~ 8e-5 추천
# depth 12: lr=3e-5 ~ 5e-5 추천
```

### Warmup Step 증가
```python
--warmup_step 5000  # 기본 2500에서 증가
```

### Gradient Clipping 강화
```python
grad_clip: float = 0.5  # 기본 1.0에서 감소
```

## 실험 재시작 가이드

```bash
# Depth 8 재실험 (보수적 설정)
export CUDA_VISIBLE_DEVICES=0
depth=8
lr=5e-5
warmup_step=5000

python train.py \
    --save_dir experiments/baseline_depth${depth}_lr${lr}_prenorm \
    --depth ${depth} \
    --lr ${lr} \
    --warmup_step ${warmup_step} \
    --batch_size 32 \
    --final_epoch 160 \
    --eval_epoch 5 \
    --valid_num_samples 512 \
    --model_type transformer \
    --reset_save_dir True
```

## 모니터링 체크리스트
1. ✓ Gradient norm이 10.0 이하로 유지되는지 확인
2. ✓ Scaler가 너무 자주 조정되지 않는지 확인 (안정적이면 65536.0 유지)
3. ✓ Loss가 갑자기 spike하지 않는지 확인
4. ✓ Validation WER이 점진적으로 감소하는지 확인

## 참고 문헌
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- GPT-3에서도 Pre-LN 사용
- T5, BERT 등 대부분의 최신 Transformer 모델들이 Pre-LN 또는 유사 구조 사용
