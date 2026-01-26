import torch
import torch.nn as nn
import sentencepiece as spm
import glob
import os
from tqdm import tqdm

def make_bpe_model(
        sentence_list,
        src_dir: str="./data/STOP_text",         
        target_dir: str="./exp",
        num_bpe: int=650,
        mask_token: str='[MASK]',
        pad_token: str='[PAD]',
        null_token: str='▁',
        blank_token: str='<blk>',
        ) -> spm.SentencePieceProcessor:
    """
    Make BPE model for STOP dataset.
    Args:
        sentence_list (list): List containing sentences.
        src_dir (str): Directory containing STOP text files.
        num_bpe (int): Number of BPE tokens.
        mask_token (str): Token for masking.
        pad_token (str): Token for padding.
        null_token (str): Token representing null space.
    Returns:
        spm.SentencePieceProcessor: Trained BPE model.
    """    
    SRC_DIR = src_dir
    NUM_BPE = num_bpe
    MASK = mask_token
    PAD = pad_token # not using now
    NULL_SPACE = null_token
    BLANK = blank_token
    
    #files = glob.glob(f"{SRC_DIR}/low.*.asr") + glob.glob(f"{SRC_DIR}/low.*.slu")
    #input_str = ",".join(files)

    tags = [ line.strip() for line in open(f"{SRC_DIR}/tag.sorted") if line.strip()]
    tags = [ NULL_SPACE + tag[1:] if tag.startswith(NULL_SPACE) else tag for tag in tags ]
    if ']' or NULL_SPACE + ']' not in tags:
        tags = tags + [ NULL_SPACE + ']' ]
    tags.sort(key=len, reverse=True)
    
    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter(sentence_list),
        model_prefix=f"{target_dir}/bpe_{NUM_BPE}",
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
    STOP_DIR = "./data/STOP_text"
    CACHE_DIR = "hubert_deberta_cache"
    TGT_DIR = "./bpe_model"
    NUM_BPE = 650
    files = glob.glob(os.path.join(CACHE_DIR, "**", "*.pt"),
                      recursive=True)    
    sentences = []        
    for file in tqdm(files):
        data = torch.load(file, map_location="cpu")
        gt = data["ground_truth"]
        dnseql = data["decoupled_normalized_seqlogical"]
        sentences.append(gt)
        sentences.append(dnseql)    

    sp = make_bpe_model(
        sentence_list=iter(sentences),
        src_dir=STOP_DIR,
        target_dir=TGT_DIR,
        num_bpe=NUM_BPE,
    )
     