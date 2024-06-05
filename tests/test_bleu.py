import pytest
from utils import compute_bleu
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
import numpy as np


class TestBLUE:
    @pytest.fixture(autouse=True)
    def pre_run_test(self):
        self.word = 1

    
    def test_compute_blue(self):
        candidate = "I am muqi li"
        references = ["I am muqi li", "I am li muqi"]
        blue = compute_bleu(candidate, references)
        assert blue == 1.0
        

    def test_compute_blue_with_single_reference(self):
        candidate = "I am muqi li"
        reference = ["I am muqi li"]
        blue = compute_bleu(candidate, reference)
        assert blue == 1.0

    def test_compute_blue_comp_sacrebleu(self):
        candidate = "I am muqi li"
        reference = ["I am muqi li hello world"]
        custom_bleu = compute_bleu(candidate, reference)
        sacrebleu_ = sacrebleu.corpus_bleu([candidate], [[ref] for ref in reference]).score
        nltk_bleu = sentence_bleu([ref.split() for ref in reference], candidate.split())
        
        # Check if scores are extremely close
        assert np.isclose(custom_bleu * 100, sacrebleu_, atol=1e-6), \
            f"Custom BLEU score {custom_bleu * 100} is not close to sacrebleu score {sacrebleu_}"
        assert np.isclose(custom_bleu, nltk_bleu, atol=1e-6), \
            f"Custom BLEU score {custom_bleu} is not close to nltk BLEU score {nltk_bleu}"

        print(custom_bleu, sacrebleu_, nltk_bleu)
