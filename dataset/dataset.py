import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, List
from utils import read_corpus
from vocab import Vocab
from functools import partial


class Seq2SeqDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str) -> None:
        src_data = read_corpus(filepath=src_path, source="src")
        tgt_data = read_corpus(filepath=tgt_path, source="tgt")
        assert len(src_data) == len(tgt_data)
        self.data = list(zip(src_data, tgt_data))

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def collate_fn(batch: List[Any], tokenizer: Vocab):
    batch.sort(key=lambda sents: len(sents[0]), reverse=True)
    
    src_sents = [item[0] for item in batch]
    tgt_sents = [item[1] for item in batch]
    src_lengths = [len(item) for item in src_sents]
    src_sent_ids_tensor = tokenizer.src.to_input_tensor(src_sents, device="cpu")
    tgt_sent_ids_tensor = tokenizer.tgt.to_input_tensor(tgt_sents, device="cpu")
    return (src_sent_ids_tensor, src_lengths), tgt_sent_ids_tensor


def get_dataloader(dataset: Seq2SeqDataset, tokenizer: Vocab, batch_size: int, shuffle=False):
    collate = partial(collate_fn, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
