import pytest
import sys
sys.path.append("../")
from utils import pad_sents


class TestVocabEntry:
    @pytest.fixture(autouse=True)
    def pre_run_test(self):
        self.word = 1
    
    def test_pad_sents(self):
        word_ids = [[1, 21, 32], [3], [1223, 32, 21, 3, 50]]
        pad_id = 0
        expected_output = [[1, 21, 32, 0, 0], [3, 0, 0, 0, 0], [1223, 32, 21, 3, 50]]
        assert pad_sents(word_ids, pad_id) == expected_output

    def test_pad_sents_all_empty_sentences(self):
        word_ids = [[], [], []]
        pad_id = 0
        expected_output = [[], [], []]
        assert pad_sents(word_ids, pad_id) == expected_output
        
    def test_convert(self):
        pass
