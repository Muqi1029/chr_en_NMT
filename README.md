# chr_en_NMT

A Neural Machine Translation implementation in PyTorch that also serves as a template for building Seq2Seq models.

## Project Structure

```
./
├── checkpoint
├── chr_en_data
│   ├── dev.chr
│   ├── dev.en
│   ├── test.chr
│   ├── test.en
│   ├── train.chr
│   └── train.en
├── dataset
│   ├── dataset.py
├── decode.py
├── models
│   ├── lstm_seq2seq.py
├── pytest.ini
├── README.md
├── results
├── run.py
├── scripts
│   ├── decode.sh
│   ├── generate_vocab.sh
│   └── run.sh
├── src.model
├── src.vocab
├── tests
│   ├── __init__.py
│   ├── test_bleu.py
│   ├── test_dataset.py
│   ├── test.ipynb
│   ├── test_model.py
│   └── test_vocabEntry.py
├── tgt.model
├── tgt.vocab
├── utils.py
├── vocab.json
└── vocab.py
```

## Usage

1. generate sentence piece model

```sh
./scripts/generate_vocab.sh
```

2. train

```sh
./scripts/run.sh
```

3. decode

```sh
./scripts/decode.sh
```

## Reference

- [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/): Stanford course taught by Christopher Manning and Richard Socher. The project structure and organization draw inspiration from the assignments and materials covered in this course.
