import random
import shutil
import time
import numpy as np
import os
import torch
import math
from typing import Any
from torch.utils.data import Dataset
from typing import List
import sentencepiece as spm
from collections import Counter


def spm_decode(value: List[str]) -> str:
    sp = spm.SentencePieceProcessor()
    sp.load('tgt.model')  # Make sure to have the 'spm.model' file in your working directory
    decoded_str = sp.decode(value)
    return decoded_str


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    if shuffle:
        random.shuffle(shuffle)
    for i in range(batch_num):
        indices = index_array[i * batch_size: max((i + 1) * batch_size, len(data))]
        couples = [data[idx] for idx in indices]
        couples.sort(key=lambda sent: len(sent[0]), reverse=True)
        src_sents = [c[0] for c in couples]
        tgt_sents = [c[1] for c in couples]
        yield src_sents, tgt_sents


def pad_sents(word_ids: List, pad_id: int):
    """ Pad list of word ids with pad_id up to max word length.
    @param word_ids (List[int]): list of word ids
    @param pad_id (int): padding id
    @returns sents_t (List[int]): list of padded word ids
    """
    max_word_len = max(len(word_id) for word_id in word_ids)
    sents_t = []
    for sent in word_ids:
        sents_t.append(sent + [pad_id] * (max_word_len - len(sent)))
    return sents_t


def compute_bleu(candidate: str, references: List[str], weight=[0.25] * 4) -> float:
    def compute_ngram(sentence, n:int):
        counter = Counter()
        for i in range(len(sentence) - n + 1):
            counter[tuple(sentence[i: i + n])] += 1
        return counter
    assert sum(weight) == 1, f"Weight must be sum up to 1"
    # compute the modified precision list
    p = []
    for i in range(4):
        candidate_counter = compute_ngram(candidate.split(), n=i+1)
        ref_counters = [compute_ngram(sent.split(), n=i+1) for sent in references]
        numerator = 0
        for n_gram in candidate_counter:
            count = candidate_counter[n_gram]
            max_ref_count = max(ref_counter[n_gram] for ref_counter in ref_counters)
            numerator += min(max_ref_count, count)
        denominator = sum(candidate_counter.values())
        p_i = numerator / denominator if denominator != 0 else 0
        p.append(p_i)

    # brevity penalty
    len_candidate = len(candidate.split())
    length_refs = np.array([len(sent.split()) for sent in references])
    ind = np.argmin(np.abs(length_refs - len_candidate), axis=0)

    BP = 1 if len_candidate >= length_refs[ind] else np.exp(1 - length_refs[ind] / len_candidate)
    print(BP)
    
    # Compute BLEU score
    if min(p) > 0:
        bleu_score = BP * math.exp(sum(weight[i] * math.log(p[i]) for i in range(4)))
    else:
        bleu_score = 0
    return bleu_score 
    
            
def read_corpus(filepath: str, source: str, test=False) -> List[str]:
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load(f"{source}.model")
    with open(filepath, "r", encoding='utf-8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            if source == 'tgt' and not test:
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)
    return data


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    

class AverageMeter:
    """Computes and stores the average and current value
    """
    def __init__(self, name, fmt=":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val, n :int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
     

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, cur_batch):
        entries = [self.prefix + self.batch_fmtstr.format(cur_batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        # {prefix} {[cur_batch/num_batches]} {meter1} {meter2}

    def _get_batch_fmtstr(self, num_batches):
        # [ cur_batch / num_batches]
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')

    progress = ProgressMeter(
        len(train_loader), 
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    model.train()
    
    end = time.time()
    
    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        print(images[0].size(), images[1].size(), i)
        
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        
        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        
        losses.update(loss.item(), n=images[0].size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
       

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

