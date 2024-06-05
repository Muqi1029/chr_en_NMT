import torch
from torch import nn
import argparse
from tqdm import tqdm
from utils import compute_bleu, read_corpus, spm_decode
from vocab import Vocab
from models.lstm_seq2seq import LstmNMT
import os
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="decode value")
    parser.add_argument("--model_path", type=str,
                        default="checkpoint/model.params")
    parser.add_argument("--input_src", type=str,
                        default="chr_en_data/test.chr")
    parser.add_argument("--input_tgt", type=str, default="chr_en_data/test.en")
    parser.add_argument("--vocab_path", type=str, default="vocab.json")
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--max_decoding_timestamp", type=int, default=80)
    parser.add_argument("--decode_path", type=str, default="results/decode.txt")

    return parser.parse_args(args)


def decode():
    args = parse_args()
    tokenizer = Vocab.load(args.vocab_path)
    test_data_src = read_corpus(args.input_src, source='src')

    model = LstmNMT.load(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    model.to(device)
    hypotheses = []
    model.eval()
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc="Decoding..."):
            src = tokenizer.src.words2indices(src_sent)
            src = torch.tensor([src], dtype=torch.long, device=device)
            hypo = model.beam_search(src, 
                                     [len(src_sent)],
                                     beam_size=args.beam_size,
                                     max_decoding_time_step=args.max_decoding_timestamp)
            hypotheses.append(hypo)

    os.makedirs(os.path.dirname(args.decode_path), exist_ok=True)
    with open(args.decode_path, "w") as f:
        top_hypotheses = [spm_decode(hypo[0].value) for hypo in hypotheses]
        # print(type(top_hypotheses[0]))
        if args.input_tgt:
            test_data_tgt = read_corpus(args.input_tgt, "tgt", test=True)
            bleu = np.mean([compute_bleu(hypo, [" ".join(tgt)]) for hypo, tgt in zip(top_hypotheses, test_data_tgt)])
            print(f"Corpus BLEU: {bleu}")
        for hypo in top_hypotheses:
            f.write(hypo + "\n")


if __name__ == '__main__':
    decode()
