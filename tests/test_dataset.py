import pytest
from dataset.dataset import Seq2SeqDataset, get_dataloader
from torch.utils.data import DataLoader
from utils import batch_iter
from vocab import Vocab



class TestDataset:
    @pytest.fixture(autouse=True)
    def pre_run_test(self):
        self.src_file_path = "chr_en_data/train.chr"
        self.tgt_file_path = "chr_en_data/train.en"
        self.batch_size = 32
        self.tokenizer = Vocab.load(file_path="vocab.json")

    def test_dataset(self):
        dataset = Seq2SeqDataset(self.src_file_path, self.tgt_file_path)
        print(len(dataset), dataset[0])
    
    def test_dataloader(self):
        dataset = Seq2SeqDataset(self.src_file_path, self.tgt_file_path) 
        dataloader = get_dataloader(dataset, self.tokenizer, self.batch_size, shuffle=True)
        for (batch_src, batch_src_lengths), batch_tgt in dataloader:
            print(batch_src.shape, batch_tgt.shape)
            print(batch_src, batch_tgt)
            print(batch_src_lengths)
            break
        
