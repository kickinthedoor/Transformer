import os
import re
import glob
import pickle
import tarfile
import urllib.request
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


# --- IWSLT14 de-en (phase 2) ---------------------------------------------
#
# Replicates the data side of fairseq's canonical prepare-iwslt14.sh recipe
# (https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-iwslt14.sh)
# so the resulting train/valid/test split is the exact one virtually every
# published IWSLT14 de-en benchmark number is computed against — same source
# archive, same tag/length filtering, same split logic (valid = every 23rd
# training line; test = dev2010 + TEDX.dev2012 + tst2010 + tst2011 + tst2012
# concatenated).
#
# Deliberate deviation from the original recipe: no Moses tokenizer, no
# lowercasing, and this project's own SentencePiece BPE instead of
# subword-nmt. sacrebleu scores on detokenized text regardless of a model's
# internal subword scheme, so this doesn't affect comparability of the final
# BLEU number — it does mean scoring stays case-sensitive ("cased" BLEU),
# which is the more modern convention anyway.

IWSLT14_URL = "http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
IWSLT14_TEST_FILE_STEMS = [
    "IWSLT14.TED.dev2010.de-en",
    "IWSLT14.TEDX.dev2012.de-en",
    "IWSLT14.TED.tst2010.de-en",
    "IWSLT14.TED.tst2011.de-en",
    "IWSLT14.TED.tst2012.de-en",
]


def download_iwslt14_archive(data_dir: str = "./data_iwslt14") -> str:
    """Downloads and extracts the fairseq-hosted IWSLT14 de-en archive (the
    same source nearly every published benchmark pipeline pulls from).
    Returns the path to the extracted `de-en` folder."""
    os.makedirs(data_dir, exist_ok=True)
    extracted_dir = os.path.join(data_dir, "de-en")

    if os.path.isdir(extracted_dir):
        print(f"[data_prep] IWSLT14 archive already extracted at '{extracted_dir}'. Skipping download.")
        return extracted_dir

    archive_path = os.path.join(data_dir, "de-en.tgz")
    if not os.path.exists(archive_path):
        print(f"[data_prep] Downloading IWSLT14 de-en archive from '{IWSLT14_URL}'...")
        urllib.request.urlretrieve(IWSLT14_URL, archive_path)

    print(f"[data_prep] Extracting '{archive_path}'...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)

    return extracted_dir


def _clean_train_tags_file(path: str) -> List[str]:
    """Strips the TED-talk XML-ish tags from a raw train.tags.de-en.{de,en}
    file: drops <url>/<talkid>/<keywords> lines entirely, keeps the text
    content of <title>/<description> tags. Mirrors the grep/sed steps in
    fairseq's prepare-iwslt14.sh.

    Critically, this must NOT drop lines that end up blank after detagging —
    doing so desyncs this file's line count from its parallel counterpart
    whenever the two languages don't have identically-empty title/description
    fields for the same talk, silently misaligning every pair from that point
    on. The original script never drops such lines either; blank pairs get
    caught later by the length filter instead, applied to both sides at once."""
    cleaned = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<url>") or line.startswith("<talkid>") or line.startswith("<keywords>"):
                continue
            line = re.sub(r"</?title>", "", line)
            line = re.sub(r"</?description>", "", line)
            cleaned.append(clean_text(line))
    return cleaned


def _parse_seg_file(path: str) -> List[str]:
    """Extracts sentence text from a dev/test <seg id="N">...</seg> file.
    These releases aren't reliably well-formed XML, so this mirrors the
    original recipe's line-based grep/sed extraction rather than using an
    XML parser."""
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '<seg id' not in line:
                continue
            text = re.sub(r'<seg id="\d+">\s*', '', line)
            text = re.sub(r'\s*</seg>\s*$', '', text)
            text = text.replace("’", "'")  # typographic apostrophe -> straight
            segs.append(clean_text(text))
    return segs


def _length_ratio_filter(
    src_lines: List[str], tgt_lines: List[str],
    ratio: float = 1.5, min_len: int = 1, max_len: int = 175
) -> Tuple[List[str], List[str]]:
    """Approximates Moses' clean-corpus-n.perl length/ratio filter: both
    sides must be within [min_len, max_len] words, and neither side may be
    more than `ratio`x longer than the other. Word counts are computed on
    simple whitespace splitting rather than Moses-tokenized text, consistent
    with this project's own SentencePiece pipeline instead of the original
    Moses+subword-nmt toolchain."""
    kept_src, kept_tgt = [], []
    for s, t in zip(src_lines, tgt_lines):
        s_len, t_len = len(s.split()), len(t.split())
        if s_len < min_len or t_len < min_len or s_len > max_len or t_len > max_len:
            continue
        if max(s_len, t_len) / min(s_len, t_len) > ratio:
            continue
        kept_src.append(s)
        kept_tgt.append(t)
    return kept_src, kept_tgt


def prepare_iwslt14_raw(data_dir: str = "./data_iwslt14") -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Downloads, cleans, and splits IWSLT14 de-en into the exact canonical
    train/valid/test partition (see module docstring above). Returns cleaned,
    untokenized sentence lists: (train_de, train_en, val_de, val_en, test_de, test_en)."""
    extracted_dir = download_iwslt14_archive(data_dir)
    lang_dir = os.path.join(extracted_dir, "de-en")
    if not os.path.isdir(lang_dir):
        lang_dir = extracted_dir  # some archive layouts extract flat

    print("[data_prep] Cleaning IWSLT14 train.tags files...")
    train_de_all = _clean_train_tags_file(os.path.join(lang_dir, "train.tags.de-en.de"))
    train_en_all = _clean_train_tags_file(os.path.join(lang_dir, "train.tags.de-en.en"))

    print(f"[data_prep] Filtering by length ratio ({len(train_de_all)} raw pairs)...")
    train_de_all, train_en_all = _length_ratio_filter(train_de_all, train_en_all)
    print(f"[data_prep] {len(train_de_all)} pairs remain after filtering.")

    # Canonical split: every 23rd line -> valid, the rest -> train (1-indexed,
    # matching awk's NR%23==0 / NR%23!=0 in the original script).
    train_de, train_en, val_de, val_en = [], [], [], []
    for i, (s, t) in enumerate(zip(train_de_all, train_en_all), start=1):
        if i % 23 == 0:
            val_de.append(s)
            val_en.append(t)
        else:
            train_de.append(s)
            train_en.append(t)

    print("[data_prep] Building the canonical test set (dev2010 + TEDX.dev2012 + tst2010/11/12)...")
    test_de, test_en = [], []
    for stem in IWSLT14_TEST_FILE_STEMS:
        de_matches = glob.glob(os.path.join(lang_dir, f"{stem}.de.xml"))
        en_matches = glob.glob(os.path.join(lang_dir, f"{stem}.en.xml"))
        if not de_matches or not en_matches:
            raise FileNotFoundError(f"Expected test files for '{stem}' not found under '{lang_dir}'.")
        test_de.extend(_parse_seg_file(de_matches[0]))
        test_en.extend(_parse_seg_file(en_matches[0]))

    print(
        f"[data_prep] IWSLT14 de-en ready: {len(train_de)} train / "
        f"{len(val_de)} valid / {len(test_de)} test pairs."
    )
    return train_de, train_en, val_de, val_en, test_de, test_en


def prepare_data_iwslt14(config: dict, force_rebuild: bool = False):
    """Master pipeline for phase 2, mirroring prepare_data() above but
    sourced from the canonical IWSLT14 de-en split instead of the IWSLT17
    HuggingFace loader. Downloads/cleans/splits -> trains tokenizers ->
    tokenizes -> caches to disk."""
    data_dir = config['paths'].get('data_dir', './data_iwslt14')
    cache_dir = config['paths'].get('cache_dir', './cache_iwslt14')
    vocab_size = config['data']['vocab_size']

    cache_path = os.path.join(cache_dir, "processed_token_data.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path) and not force_rebuild:
        print(f"[data_prep] Loading pre-tokenized IWSLT14 dataset from cache '{cache_path}'...")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        de_tok, en_tok = setup_tokenizers(
            train_de_path=cached['train_de_path'], train_en_path=cached['train_en_path'],
            vocab_size=vocab_size,
            de_prefix=config['paths'].get('src_model_prefix', 'de_bpe'),
            en_prefix=config['paths'].get('tgt_model_prefix', 'en_bpe'),
        )
        return (
            cached['train_de'], cached['train_en'],
            cached['val_de'], cached['val_en'],
            cached['test_de'], cached['test_en'],
            de_tok, en_tok,
            cached['train_src_raw'], cached['train_tgt_raw']
        )

    train_src_raw, train_tgt_raw, val_src_raw, val_tgt_raw, test_src_raw, test_tgt_raw = prepare_iwslt14_raw(data_dir)

    # SentencePiece needs plain text files to train on, not in-memory lists.
    train_de_path = os.path.join(data_dir, "train.de")
    train_en_path = os.path.join(data_dir, "train.en")
    with open(train_de_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_src_raw) + "\n")
    with open(train_en_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_tgt_raw) + "\n")

    de_tok, en_tok = setup_tokenizers(
        train_de_path=train_de_path,
        train_en_path=train_en_path,
        vocab_size=vocab_size,
        de_prefix=config['paths'].get('src_model_prefix', 'de_bpe'),
        en_prefix=config['paths'].get('tgt_model_prefix', 'en_bpe'),
    )

    print(f"[data_prep] Tokenizing {len(train_src_raw)} training pairs...")
    train_de = [de_tok.encode(s) for s in train_src_raw]
    train_en = [en_tok.encode(t) for t in train_tgt_raw]

    print(f"[data_prep] Tokenizing {len(val_src_raw)} validation pairs...")
    val_de = [de_tok.encode(s) for s in val_src_raw]
    val_en = [en_tok.encode(t) for t in val_tgt_raw]

    print(f"[data_prep] Tokenizing {len(test_src_raw)} test pairs...")
    test_de = [de_tok.encode(s) for s in test_src_raw]
    test_en = [en_tok.encode(t) for t in test_tgt_raw]

    with open(cache_path, 'wb') as f:
        pickle.dump({
            'train_de': train_de, 'train_en': train_en,
            'val_de': val_de, 'val_en': val_en,
            'test_de': test_de, 'test_en': test_en,
            'train_src_raw': train_src_raw, 'train_tgt_raw': train_tgt_raw,
            'train_de_path': train_de_path, 'train_en_path': train_en_path,
        }, f)
    print(f"[data_prep] Saved processed IWSLT14 dataset cache to '{cache_path}'.")

    return train_de, train_en, val_de, val_en, test_de, test_en, de_tok, en_tok, train_src_raw, train_tgt_raw