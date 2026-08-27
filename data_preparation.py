import os
import re
import pickle
from typing import Tuple, List
from datasets import load_dataset
from tokenizer import SentencePieceTokenizer


def download_iwslt17_dataset(data_dir: str = "./data") -> Tuple[str, str, str, str, str, str]:
    """
    Downloads IWSLT2017 TED Talks (de-en) train, validation, and test splits using HuggingFace Datasets and saves them as raw text files into data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)

    raw_train_de, raw_train_en = os.path.join(data_dir, "train.de"), os.path.join(data_dir, "train.en")
    raw_val_de, raw_val_en     = os.path.join(data_dir, "val.de"), os.path.join(data_dir, "val.en")
    raw_test_de, raw_test_en   = os.path.join(data_dir, "test.de"), os.path.join(data_dir, "test.en")

    # Skip download if all raw files already exist
    if all(os.path.exists(p) for p in [raw_train_de, raw_val_de, raw_test_de]):
        print(f"[data_prep] Raw IWSLT17 files found in '{data_dir}'. Skipping download.")
        return raw_train_de, raw_train_en, raw_val_de, raw_val_en, raw_test_de, raw_test_en

    print(f"[data_prep] Downloading IWSLT2017 TED Talks (de-en) dataset to '{data_dir}'...")
    dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-de-en")

    # Helper function to write splits
    def write_split(split_name: str, out_de: str, out_en: str):
        print(f"[data_prep] Writing {os.path.basename(out_de)} and {os.path.basename(out_en)}...")
        with open(out_de, "w", encoding="utf-8") as f_de, open(out_en, "w", encoding="utf-8") as f_en:
            for example in dataset[split_name]:
                f_de.write(example["translation"]["de"].strip() + "\n")
                f_en.write(example["translation"]["en"].strip() + "\n")

    # Write Train, Validation, and Test
    write_split("train", raw_train_de, raw_train_en)
    write_split("validation", raw_val_de, raw_val_en)
    write_split("test", raw_test_de, raw_test_en)

    print("[data_prep] Download and extraction complete!")
    return raw_train_de, raw_train_en, raw_val_de, raw_val_en, raw_test_de, raw_test_en


def clean_text(text: str) -> str:
    """Basic text normalization."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def filter_and_clean_pairs(src_file: str, tgt_file: str, max_len: int = 100, is_test: bool = False) -> Tuple[List[str], List[str]]:
    """
    Reads raw files, cleans lines, and removes pairs exceeding max_len words.
    For test sets, length filtering is usually skipped (is_test=True) to evaluate standard benchmarks.
    """
    cleaned_src, cleaned_tgt = [], []

    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt:

        for line_src, line_tgt in zip(f_src, f_tgt):
            src_str = clean_text(line_src)
            tgt_str = clean_text(line_tgt)

            if is_test:
                # Do not drop test set lines—keep all non-empty lines for accurate benchmark evaluation
                if len(src_str) > 0 and len(tgt_str) > 0:
                    cleaned_src.append(src_str)
                    cleaned_tgt.append(tgt_str)
            else:
                # Filter long training/val pairs to save memory and avoid OOM
                if 0 < len(src_str.split()) <= max_len and 0 < len(tgt_str.split()) <= max_len:
                    cleaned_src.append(src_str)
                    cleaned_tgt.append(tgt_str)

    return cleaned_src, cleaned_tgt


def setup_tokenizers(
    train_de_path: str,
    train_en_path: str,
    vocab_size: int = 32000,
    de_prefix: str = "de_bpe",
    en_prefix: str = "en_bpe"
) -> Tuple[SentencePieceTokenizer, SentencePieceTokenizer]:
    """Trains SentencePiece BPE tokenizers if .model files don't exist, then loads them."""
    de_model_file = f"{de_prefix}.model"
    en_model_file = f"{en_prefix}.model"

    # Ensure output directories exist before training SentencePiece
    de_dir = os.path.dirname(de_prefix)
    if de_dir:
        os.makedirs(de_dir, exist_ok=True)
        
    en_dir = os.path.dirname(en_prefix)
    if en_dir:
        os.makedirs(en_dir, exist_ok=True)

    de_tok = SentencePieceTokenizer(vocab_size=vocab_size)
    en_tok = SentencePieceTokenizer(vocab_size=vocab_size)

    if not os.path.exists(de_model_file):
        print(f"[data_prep] Training German SentencePiece model ('{de_prefix}')...")
        de_tok.train_model(train_de_path, de_prefix)

    if not os.path.exists(en_model_file):
        print(f"[data_prep] Training English SentencePiece model ('{en_prefix}')...")
        en_tok.train_model(train_en_path, en_prefix)

    de_tok.load_model(de_model_file)
    en_tok.load_model(en_model_file)

    return de_tok, en_tok


def prepare_data(
    config: dict,
    force_rebuild: bool = False
):
    """
    Master pipeline:
    Downloads train/val/test splits -> Trains Tokenizers -> Tokenizes -> Caches to disk.
    """
    data_dir = config['paths'].get('data_dir', './data')
    cache_dir = config['paths'].get('cache_dir', './cache')
    max_len = config['data']['max_seq_len']
    vocab_size = config['data']['vocab_size']

    # Step 1: Download raw data (train, val, test)
    raw_train_de, raw_train_en, raw_val_de, raw_val_en, raw_test_de, raw_test_en = download_iwslt17_dataset(data_dir)

    # Step 2: Setup Tokenizers
    de_tok, en_tok = setup_tokenizers(
        train_de_path=raw_train_de,
        train_en_path=raw_train_en,
        vocab_size=vocab_size,
        de_prefix=config['paths'].get('src_model_prefix', 'de_bpe'),
        en_prefix=config['paths'].get('tgt_model_prefix', 'en_bpe')
    )

    # Step 3: Check cached token arrays
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "processed_token_data.pkl")

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"[data_prep] Loading pre-tokenized dataset from cache '{cache_path}'...")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return (
            cached['train_de'], cached['train_en'],
            cached['val_de'], cached['val_en'],
            cached['test_de'], cached['test_en'],
            de_tok, en_tok,
            cached['train_src_raw'], cached['train_tgt_raw']
        )

    # Step 4: Process raw text if cache isn't present
    print("[data_prep] Cleaning & filtering parallel sentences...")
    train_src_raw, train_tgt_raw = filter_and_clean_pairs(raw_train_de, raw_train_en, max_len, is_test=False)
    val_src_raw, val_tgt_raw     = filter_and_clean_pairs(raw_val_de, raw_val_en, max_len, is_test=False)
    test_src_raw, test_tgt_raw   = filter_and_clean_pairs(raw_test_de, raw_test_en, max_len, is_test=True)

    print(f"[data_prep] Tokenizing {len(train_src_raw)} training pairs...")
    train_de = [de_tok.encode(s) for s in train_src_raw]
    train_en = [en_tok.encode(t) for t in train_tgt_raw]

    print(f"[data_prep] Tokenizing {len(val_src_raw)} validation pairs...")
    val_de = [de_tok.encode(s) for s in val_src_raw]
    val_en = [en_tok.encode(t) for t in val_tgt_raw]

    print(f"[data_prep] Tokenizing {len(test_src_raw)} test pairs...")
    test_de = [de_tok.encode(s) for s in test_src_raw]
    test_en = [en_tok.encode(t) for t in test_tgt_raw]

    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'train_de': train_de, 'train_en': train_en,
            'val_de': val_de, 'val_en': val_en,
            'test_de': test_de, 'test_en': test_en,
            # Raw (untokenized) train text, needed for on-the-fly BPE-dropout
            # tokenization during training — val/test stay deterministic, so they
            # don't need their raw text cached.
            'train_src_raw': train_src_raw, 'train_tgt_raw': train_tgt_raw
        }, f)
    print(f"[data_prep] Saved processed dataset cache to '{cache_path}'.")

    return train_de, train_en, val_de, val_en, test_de, test_en, de_tok, en_tok, train_src_raw, train_tgt_raw