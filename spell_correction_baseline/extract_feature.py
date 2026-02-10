import os, glob
import torch
import librosa
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import Wav2Vec2FeatureExtractor, HubertForCTC, AutoProcessor
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import unicodedata
from pyctcdecode import build_ctcdecoder

TARGET_SR = 16000

def normalize_text(s: str) -> str:
    # 1) 유니코드 정규화(특수 apostrophe 같은 거 통일)
    s = unicodedata.normalize("NFKC", s)

    # 2) 대문자
    s = s.upper()

    # 3) 줄바꿈/탭 -> 공백
    s = re.sub(r"\s+", " ", s)

    # 4) 알파벳/숫자/공백만 남기고 나머지 제거
    #    (따옴표, 마침표, 물음표, 콤마, 콜론, 하이픈, 슬래시 등 전부 제거됨)
    s = re.sub(r"[^A-Z0-9\' ]+", "", s)

    # 5) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_stop_manifest(manifest_path,
                       train_path="train.tsv",
                       test_path="test.tsv",
                       eval_path="eval.tsv") -> Dict:
    sep = '\t'
    # test_Set

    sets = [train_path, eval_path, test_path]
    rets = [ {} for _ in range(len(sets)) ]
    #sets = [eval_path, test_path]
    #rets = [eval_dict, test_dict]
    assert len(sets) == len(rets)
    skip_cases = 0
    for s in range(len(sets)):
        with open(os.path.join(manifest_path, sets[s]), "r", encoding="utf-8") as f:
            # the first line (separator: \t)
            # file_id, domain, gender, native, utterance,  \\
            # seqlogical, normalized_utterance,  \\
            # normalized_seqlogical, decoupled_normalized_seqlogical
            field = f.readline().strip().split(sep)
            for i in range(len(field)):
                rets[s].setdefault(field[i], [])

            for line in f.readlines():
                item = line.strip().split(sep)
                if len(field) != len(item):
                    skip_cases += 1
                    continue

                for j in range(len(field)):
                    rets[s][field[j]].append(item[j])

            print(f"{skip_cases=}")

    manifest = {}
    for i in range(len(sets)):
        key = str(Path(sets[i]).with_suffix(""))
        print(f"{key=}")
        manifest[key] = rets[i]

    return manifest

def load_audio_16k(file_path, target_sr=TARGET_SR):
    """
    librosa.load(..., sr=target_sr)는 '가정'이 아니라
    로드 후 target_sr로 리샘플링까지 수행합니다.
    """
    y, sr = librosa.load(file_path, sr=target_sr)
    return y  # 1D np.array

def parse_digit_label_from_filename(path: str) -> int:
    # e.g., "1_xxxx_pp.wav" -> 1
    base = os.path.basename(path)
    digit_str = base.split("_")[0]
    return int(digit_str)

@torch.no_grad()
def extract_and_save_hubert_features(
    manifest,
    cache_dir,
    model_name="facebook/hubert-large-ls960-ft",
    text_model_name="microsoft/deberta-v3-large",    
    batch_size=32,
    sr=16000,
    device="cuda",
    audio_file_prefix="./data/stop",
    manifest_type="train",
    n_best=4,
):
    os.makedirs(cache_dir, exist_ok=True)
    wav_files = manifest["file_id"]

    # for audio feature
    # HF 권장: feature_extractor는 padding/attn_mask를 깔끔하게 만들어줌
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertForCTC.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # for text feature
    text_model = AutoModel.from_pretrained(text_model_name).to(device).eval()
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # 숫자 추가 지금은 안해도 될 듯??
    #new_tokens = [str(i) for i in range(10)]
    #num_added = processor.tokenizer.add_tokens(new_tokens)
    #processor_token_num = len(processor.tokenizer)

    # for CTC decoder
    vocab = processor.tokenizer.get_vocab()
    vocab_list = [token for token, idx in sorted(vocab.items(), key=lambda item: item[1])]
    decoder = build_ctcdecoder(vocab_list)
    # 메타 저장(나중에 dataset에서 인덱싱용)
    index = []  # list of dicts: {wav_path, digit, feat_path}

    for start in tqdm(range(0, len(wav_files), batch_size)):
        chunk = wav_files[start:start + batch_size]
        
        check_existing = []
        for i in range(len(chunk)):
            check_existing.append(False)
        for i, wav_path in enumerate(chunk):
            pt_path = str(Path(wav_path).with_suffix('.pt'))
            feat_path = os.path.join(cache_dir, pt_path)
            if os.path.exists(feat_path):
                check_existing[i] = True
        if all(check_existing):            
            continue
        # 오디오 파일 이름 filter
        # from: eval_0/alarm_eval_0/00002321.wav
        # to: eval_0/alarm_eval/00002321.wav
        waves = []
        paths = []
        for p in chunk:
            file_path = []
            ps = p.split("/")
            ps[1] = ps[1].replace("_1", "")
            ps[1] = ps[1].replace("_0", "")
            file_path.append(audio_file_prefix)
            for particle_path in ps: file_path.append(particle_path)
            current_path = Path(*file_path)
            wave = load_audio_16k(current_path, target_sr=sr)
            paths.append(current_path)
            waves.append(wave)

        batch = feature_extractor(
            waves,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        input_values = batch["input_values"].to(device)           # (B, T)
        attention_mask = batch["attention_mask"].to(device)       # (B, T) 1/0

        outputs = model(input_values=input_values,
                       attention_mask=attention_mask,
                       output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] # 마지막 레이어

        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        all_hyps = []
        for lp in log_probs:
            beams = decoder.decode_beams(
                lp,
                beam_width=n_best,
            )
            all_hyps.append(beams[:n_best])

        transcriptions = processor.batch_decode(pred_ids)
        ground_truth = manifest["utterance"][start:start+batch_size]
        
        for i in range(len(transcriptions)):
            transcriptions[i] = normalize_text(transcriptions[i])
            ground_truth[i] = normalize_text(ground_truth[i])

        n_best_hypotheses = []        
        for i, hyps in enumerate(all_hyps):
            hypotheses = []
            for rank, text_chunk in enumerate(hyps):                
                #print(f"  {rank+1}: {text_chunk[0]}  (score={text_chunk[-1]:.2f})")
                hypotheses.append(normalize_text(text_chunk[0]))
            n_best_hypotheses.append(hypotheses)
        
         ## TEXT feature        
        text_batch = text_tokenizer(
            transcriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,            
            return_attention_mask=True,
        ).to(device)
        
        text_outputs = text_model(**text_batch)
        text_feats = text_outputs.last_hidden_state # B, T, D
        text_feat_mask = text_batch.attention_mask
        text_feat_lens = text_feat_mask.sum(-1) # B,

        # feats용 mask 만들기: wav 길이 -> feat 길이로 변환
        feats = last_hidden_state             # (B, T_feat, D)
        wav_lens = attention_mask.sum(dim=1)  # (B,)
        feat_lens = model._get_feat_extract_output_lengths(wav_lens)  # (B,)
        T_feat = feats.size(1)
        feat_mask = (torch.arange(T_feat, device=device)[None, :] < feat_lens[:, None])  # (B, T_feat) bool

        # 샘플별 저장(랜덤 access 쉽게)
        for i, wav_path in enumerate(chunk):
            pt_path = str(Path(wav_path).with_suffix('.pt'))
            feat_path = os.path.join(cache_dir, pt_path)
            feat_dir = os.path.dirname(feat_path)
            if feat_dir:
                os.makedirs(feat_dir, exist_ok=True)
            torch.save(
                {
                    "feats": feats[i, :feat_lens[i]].detach().cpu(),            # (T_feat, D)
                    "feat_mask": feat_mask[i, :feat_lens[i]].detach().cpu(),    # (T_feat,)
                    "wav_path": wav_path,
                    "greedy_hypothesis": transcriptions[i],
                    "ground_truth": ground_truth[i],
                    "n_best_hypotheses": n_best_hypotheses[i],
                    "text_feats": text_feats[i, :text_feat_lens[i]].detach().cpu(),
                    "text_feat_mask": text_feat_mask[i, :text_feat_lens[i]].detach().cpu(),
                },
                feat_path
            )
            # 아직 잘모르겠는데, 나중을 위해 저장
            index.append({"wav_path": wav_path, "feat_path": feat_path})
        
    # 전체 인덱스도 저장
    index_path = os.path.join(cache_dir, f"{manifest_type}_index.pt")
    torch.save(index, index_path)
    print(f"Saved features to: {cache_dir}")
    print(f"Index saved to: {index_path}")
    return index_path

if __name__ == "__main__":
    # load manifest
    STOP_MANIFEST_DIR = "./data/stop/manifests"
    manifest = load_stop_manifest(STOP_MANIFEST_DIR)

    for key in manifest.keys():
        cache_dir = f"./hubert_deberta_cache_retrial"
        index_path = extract_and_save_hubert_features(
            manifest=manifest[key],
            cache_dir=cache_dir,
            batch_size=8,
            device="cuda",
            manifest_type=key,
        )
