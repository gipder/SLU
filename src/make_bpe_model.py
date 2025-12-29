import torch
import torch.nn as nn
import sentencepiece as spm
import glob

def make_bpe_model(
        src_dir: str="./data/STOP_text", 
        num_bpe: int=650,
        mask_token: str='[MASK]',
        pad_token: str='[PAD]',
        null_token: str='▁',
        blank_token: str='<blk>',
        ) -> spm.SentencePieceProcessor:
    """
    Make BPE model for STOP dataset.
    Args:
        src_dir (str): Directory containing STOP text files.
        num_bpe (int): Number of BPE tokens.
        mask_token (str): Token for masking.
        pad_token (str): Token for padding.
        null_token (str): Token representing null space.
    Returns:
        spm.SentencePieceProcessor: Trained BPE model.
    """    
    SRC_DIR=src_dir
    NUM_BPE = num_bpe
    MASK = mask_token
    PAD = pad_token # not using now
    NULL_SPACE = null_token
    BLANK = blank_token
    
    files = glob.glob(f"{SRC_DIR}/low.*.asr") + glob.glob(f"{SRC_DIR}/low.*.slu")
    input_str = ",".join(files)

    tags = [ line.strip() for line in open(f"{SRC_DIR}/tag.sorted") if line.strip()]
    tags = [ NULL_SPACE + tag[1:] if tag.startswith(NULL_SPACE) else tag for tag in tags ]
    if ']' or NULL_SPACE + ']' not in tags:
        tags = tags + [ NULL_SPACE + ']' ]
    tags.sort(key=len, reverse=True)

    spm.SentencePieceTrainer.Train(
        input=input_str,
        model_prefix=f"{SRC_DIR}/bpe_{NUM_BPE}",
        vocab_size=NUM_BPE,
        model_type="bpe",
        character_coverage=1.0,
        control_symbols=[BLANK, MASK],
        pad_id=2,
        unk_id=3,
        bos_id=4,
        eos_id=5,        
        user_defined_symbols=tags,
    )

    # to check the maximum length
    bpe_model = f"{SRC_DIR}/bpe_{NUM_BPE}.model"
    sp = spm.SentencePieceProcessor()
    sp.Load(bpe_model)

    return sp


if __name__ == "__main__":    
    STOP_DIR="./data/STOP_text"
    sp = make_bpe_model(
        src_dir=STOP_DIR,
        num_bpe=650,
    )
            
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

    #LIMIT_LEN = 256
    #max_len = max(max_len, LIMIT_LEN)
    print("tgt max seq length:", max_len)
    print("tgt min seq length:", min_len)

    print("src max seq length:", max_src_len)
    print("src min seq length:", min_src_len)
