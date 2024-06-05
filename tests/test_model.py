import pytest
from dataset.dataset import Seq2SeqDataset, get_dataloader
from models.lstm_seq2seq import LstmNMT
from vocab import Vocab
import sentencepiece as spm


class TestModel:
    @pytest.fixture(autouse=True)
    def pre_run_test(self):
        self.embed_size = 20
        self.hidden_size = 12
        self.batch_size = 12
        self.src_file_path = "chr_en_data/train.chr"
        self.tgt_file_path = "chr_en_data/train.en"
        self.tokenizer = Vocab.load(file_path="vocab.json")
        self.model = LstmNMT(embed_size=self.embed_size,
                             hidden_size=self.hidden_size,
                             tokenizer=self.tokenizer)
        self.dataset = Seq2SeqDataset(
            src_path=self.src_file_path, tgt_path=self.tgt_file_path)
        self.dataloader = get_dataloader(self.dataset,
                                         self.tokenizer,
                                         batch_size=self.batch_size,
                                         shuffle=True)

    def test_model(self):
        print(self.model.eval())

    def test_encode(self):
        for (src, src_lengths), tgt in self.dataloader:
            encoded_states, (h, c) = self.model.encode(src, src_lengths)
            assert encoded_states.shape == (self.batch_size, src.size(1), 2 * self.hidden_size)
            assert h.shape == (self.batch_size, self.hidden_size)
            assert c.shape == (self.batch_size, self.hidden_size)
            break
            

    def test_decode(self):
        for (src, src_lengths), tgt in self.dataloader:
            batch_size, seq_len = tgt.shape
            encoded_states, state = self.model.encode(src, src_lengths)
            enc_mask = self.model._generate_enc_mask(src, src_lengths)
            ps = self.model.decode(tgt, state, encoded_states, enc_mask)
            assert ps.shape == (batch_size, seq_len - 1, len(self.tokenizer.tgt)), \
                f"expected ({batch_size}, {seq_len - 1}, {len(self.tokenizer.tgt)}), but got ({ps.shape})"
            break

    def test_forward(self):
        for (src, src_lengths), tgt in self.dataloader:
            loss = self.model(src, src_lengths, tgt)
            print(loss.item())
            break
    
    def test_beam_search(self):
        i = 0
        for i, ((src, src_lengths), tgt) in enumerate(self.dataloader):
            completed_hypothesis = self.model.beam_search(src[0], src_lengths[:1])
            print(len(completed_hypothesis))
            sp = spm.SentencePieceProcessor()
            sp.load('tgt.model')  # Make sure to have the 'spm.model' file in your working directory

            # Decode the tokenized string
            decoded_str = sp.decode(completed_hypothesis[0].value)
            print("decoded str".center(50, '='))
            print(decoded_str)
            print("tgt str".center(50, "="))
            print(self.model.tokenizer.tgt.indices2words(tgt[0].detach().numpy().tolist()))
            if i == 5: break
