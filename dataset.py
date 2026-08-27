from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class WMTDataset(Dataset):
    def __init__(self, src_encoded: List[List[int]], tgt_encoded: List[List[int]]):
        # Keep raw lists to avoid Python/PyTorch IPC serialization overhead
        self.src_data = src_encoded
        self.tgt_data = tgt_encoded

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_data[idx], self.tgt_data[idx]


class BPEDropoutDataset(Dataset):
    """
    Tokenizes on-the-fly with SentencePiece sampling (BPE dropout) at every
    access, instead of serving fixed pre-tokenized ids like WMTDataset — so the
    same sentence gets a different subword segmentation across epochs. Meant
    for training data only; val/test should stay on WMTDataset's deterministic
    pre-tokenized path.
    """
    def __init__(self, src_texts: List[str], tgt_texts: List[str], src_tokenizer, tgt_tokenizer, alpha: float = 0.1):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src_ids = self.src_tokenizer.encode(self.src_texts[idx], sample=True, alpha=self.alpha)
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_texts[idx], sample=True, alpha=self.alpha)
        return src_ids, tgt_ids


def fast_collate_fn(batch, pad_id: int = 0):
    """
    Converts raw Python lists to tensors directly inside the collate step.
    """
    src_list, tgt_list = zip(*batch)

    # Convert to Tensors only at batch creation time
    src_tensors = [torch.tensor(x, dtype=torch.long) for x in src_list]
    tgt_tensors = [torch.tensor(x, dtype=torch.long) for x in tgt_list]

    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_id)

    # Simple 4D masks (1 for pad, 0 for token)
    src_mask = (src_padded == pad_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt_padded == pad_id).unsqueeze(1).unsqueeze(2)

    return (src_padded, src_mask), (tgt_padded, tgt_mask)


def create_single_dataloader(
    src_encoded: List[List[int]],
    tgt_encoded: List[List[int]],
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,     # Default to 0 to bypass Windows process spawn overhead
    pin_memory: bool = True
) -> DataLoader:
    dataset = WMTDataset(src_encoded, tgt_encoded)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=fast_collate_fn,
        num_workers=0,         # Force 0 workers for direct in-memory batching
        pin_memory=pin_memory
    )


def create_bpe_dropout_dataloader(
    src_texts: List[str],
    tgt_texts: List[str],
    src_tokenizer,
    tgt_tokenizer,
    alpha: float = 0.1,
    batch_size: int = 64,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """Training dataloader with on-the-fly BPE-dropout tokenization (see BPEDropoutDataset)."""
    dataset = BPEDropoutDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, alpha=alpha)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=fast_collate_fn,
        num_workers=0,
        pin_memory=pin_memory
    )


def get_dataloaders(
    train_src: List[List[int]],
    train_tgt: List[List[int]],
    val_src: List[List[int]],
    val_tgt: List[List[int]],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_loader = create_single_dataloader(
        train_src, train_tgt, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = create_single_dataloader(
        val_src, val_tgt, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    return train_loader, val_loader