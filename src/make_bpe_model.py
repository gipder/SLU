import torch
import torch.nn as nn
import sentencepiece as spm
import glob

NUM_BPE = 650
MASK = '[MASK]'
PAD = '[PAD]'
NULL = '▁'

STOP_DIR="./data/STOP_text"
files = glob.glob(f"{STOP_DIR}/low.*.asr") + glob.glob(f"{STOP_DIR}/low.*.slu")
input_str = ",".join(files)

tags = [ line.strip() for line in open(f"{STOP_DIR}/tag.sorted") if line.strip()]
tags = [ NULL + tag[1:] if tag.startswith(NULL) else tag for tag in tags ]
if ']' or NULL + ']' not in tags:
    tags = tags + [ NULL + ']' ]
tags.sort(key=len, reverse=True)

spm.SentencePieceTrainer.Train(
    input=input_str,
    model_prefix=f"{STOP_DIR}/bpe_{NUM_BPE}",
    vocab_size=NUM_BPE,
    model_type="bpe",
    character_coverage=1.0,
    control_symbols=[MASK],
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_id=4,
    user_defined_symbols=tags,
)

# to check the maximum length
bpe_model = f"{STOP_DIR}/bpe_{NUM_BPE}.model"
sp = spm.SentencePieceProcessor()
sp.Load(bpe_model)

#input_path = f"{STOP_DIR}/low.*.asr"
input_paths = sorted(glob.glob(f"{STOP_DIR}/low.*.asr"))
#target_path = f"{STOP_DIR}/low.*.slu"
target_paths = sorted(glob.glob(f"{STOP_DIR}/low.*.slu"))

print(f"{input_paths=}")
print(f"{target_paths=}")

max_len = 0
min_len = 10000000
max_src_len = 0
min_src_len = 10000000
for input_path, target_path in zip(input_paths, target_paths):
    with open(input_path, encoding="utf-8") as fin, \
        open(target_path, encoding="utf-8") as ftgt:

        for src_line, tgt_line in zip(fin, ftgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if not src_line or not tgt_line:
                continue

            src_ids = sp.encode(src_line, out_type=int)
            tgt_ids = sp.encode(tgt_line, out_type=int)

            max_len = max(max_len, len(tgt_ids))
            min_len = min(min_len, len(tgt_ids))
            max_src_len = max(max_src_len, len(src_ids))
            min_src_len = min(min_src_len, len(src_ids))
            """
            if len(tgt_ids) == 5:
                print(f"{min_len=} {src_line=}, {tgt_line=}")
                print(f"{tgt_ids=}")
            if len(tgt_ids) == 135:
                print(f"{max_len=} {src_line=}, {tgt_line=}")

            if len(src_ids) == 1:
                print(f"{min_src_len=} {src_line=}, {tgt_line=}")
            if len(src_ids) == 97:
                print(f"{max_src_len=} {src_line=}, {tgt_line=}")
            """

#LIMIT_LEN = 256
#max_len = max(max_len, LIMIT_LEN)
print("tgt max seq length:", max_len)
print("tgt min seq length:", min_len)

print("src max seq length:", max_src_len)
print("src min seq length:", min_src_len)
