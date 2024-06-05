import time
import torch
from torch import nn
import argparse
from torch.nn.utils import clip_grad_norm_
from dataset.dataset import Seq2SeqDataset, get_dataloader
from models.lstm_seq2seq import LstmNMT
from utils import AverageMeter, ProgressMeter
from vocab import Vocab
import numpy as np
from tqdm import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_size", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--uniform_init", type=float, default=0.1)

    parser.add_argument("--train_src", type=str,
                        default="chr_en_data/train.chr")
    parser.add_argument("--train_tgt", type=str,
                        default="chr_en_data/train.en")
    parser.add_argument("--dev_src", type=str, default="chr_en_data/dev.chr")
    parser.add_argument("--dev_tgt", type=str, default="chr_en_data/dev.en")

    parser.add_argument("--vocab", type=str, default="vocab.json")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--model_path", type=str,
                        default="checkpoint/model.params")

    parser.add_argument("--patient", type=int, default=1)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--valid_per_epoch", type=int, default=3)

    args = parser.parse_args(args)
    return args


def main():
    args = parse_args()
    # 1. get vocab
    tokenizer = Vocab.load(args.vocab)

    # 2. get datasets and dataloaders
    train_dataset = Seq2SeqDataset(
        src_path=args.train_src, tgt_path=args.train_tgt)
    train_dataloader = get_dataloader(
        train_dataset, tokenizer, batch_size=args.batch_size, shuffle=True)
    dev_dataset = Seq2SeqDataset(src_path=args.dev_src, tgt_path=args.dev_tgt)
    dev_dataloader = get_dataloader(
        dev_dataset, tokenizer, batch_size=args.batch_size, shuffle=True)

    # 3. get model
    model = LstmNMT(embed_size=args.embed_size, hidden_size=args.hidden_size,
                    tokenizer=tokenizer, dropout_rate=args.dropout)
    uniform_init = float(args.uniform_init)
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' %
              (uniform_init, uniform_init))
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model to {device}")
    model.to(device)

    # 4. train the model
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    train(model, train_dataloader, dev_dataloader, optimizer, device, args)

    # 5. save the model
    model.save(args.model_path)


def valid(model, dev_dataloader, device, args):
    is_train = model.training
    model.eval()
    print("start to evaluate model".center(50, "="))
    loss = AverageMeter("loss")
    tgt_words = AverageMeter("tgt_words")
    with torch.no_grad():
        for (src, src_lengths), tgt in tqdm(dev_dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            l = model(src, src_lengths, tgt)
            loss.update(l.item())
            tgt_words.update(sum([len(s) - 1 for s in tgt]))
        ppl = np.exp(loss.sum / tgt_words.sum)
    if is_train:
        model.train()
    return ppl


def train(model, train_dataloader, dev_dataloader, optimizer, device, args):
    model.train()
    hist_valid_scores = []
    num_trial = 0
    patience = 0
    for epoch in range(1, args.num_epochs + 1):
        time_metric = AverageMeter(name="batch_time", fmt=":.4f")
        loss_metric = AverageMeter(name="loss", fmt=":.4f")
        progress = ProgressMeter(num_batches=len(
            train_dataloader), meters=[time_metric, loss_metric],
            prefix="Epoch %d" % epoch)
        for cur_batch, ((src, src_lengths), tgt) in enumerate(train_dataloader):
            start_time = time.time()
            src = src.to(device)
            tgt = tgt.to(device)
            l = model(src, src_lengths, tgt)
            l /= len(src)
            optimizer.zero_grad()
            l.backward()
            clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            time_metric.update(time.time() - start_time)
            loss_metric.update(l.item())
            progress.display(cur_batch=cur_batch + 1)
        if epoch % args.valid_per_epoch == 0:
            # start to evaluate
            score = -valid(model, dev_dataloader, device, args=args)
            print(f"score {score} ")
            is_better = len(hist_valid_scores) == 0 or score > max(
                hist_valid_scores)
            if is_better:
                patience = 0
                model.save(args.model_path)
                torch.save(optimizer.state_dict(), args.model_path + ".optim")
            elif patience < args.patience:
                patience += 1
                print(f"hit patience {patience}")
                if patience == args.patience:
                    num_trial += 1
                    print(f"hit #{num_trial} trials")
                    if num_trial == args.max_num_trial:
                        print(f"hit max trial {args.max_num_trial}")
                        exit(0)

                    lr = optimizer.param_groups[0]['lr'] * args.lr_decay

                    # load model
                    print(
                        'load previously best model and decay learning rate to %f' % lr)
                    params = torch.load(
                        args.model_path, map_location=lambda storage, loc: storage)
                    model.load_state_dict(params['state_dict'])
                    model = model.to(args.device)

                    print('restore parameters of the optimizers')
                    optimizer.load_state_dict(
                        torch.load(args.model_path + '.optim'))

                    # set new lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    patience = 0


if __name__ == '__main__':
    main()
