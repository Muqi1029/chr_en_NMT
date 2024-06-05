from typing import List, Tuple
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from vocab import Vocab
from collections import namedtuple
import os


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class ModelEmbedding(nn.Module):
    def __init__(self, embed_size: int, tokenizer: Vocab):
        super().__init__()
        self.embed_size = embed_size
        src_pad_token_idx = tokenizer.src['<pad>']
        tgt_pad_token_idx = tokenizer.tgt['<pad>']
        self.source_embed = nn.Embedding(
            len(tokenizer.src), embedding_dim=embed_size, padding_idx=src_pad_token_idx)
        self.target_embed = nn.Embedding(
            len(tokenizer.tgt), embedding_dim=embed_size, padding_idx=tgt_pad_token_idx)


class LstmNMT(nn.Module):
    def __init__(self, embed_size, hidden_size, tokenizer: Vocab, dropout_rate: float = 0.2):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.embedding_layer = ModelEmbedding(embed_size, tokenizer)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               batch_first=True,
                               bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=embed_size + hidden_size,
                                   hidden_size=hidden_size)

        self.get_initial_h_state = nn.Linear(2 * hidden_size, hidden_size)
        self.get_initial_c_state = nn.Linear(2 * hidden_size, hidden_size)

        # this is for projecting the embedded hidden state(because of the bidirectional layer),
        # and then compute the attention weight using dot product between decoded hidden state and the projected embedded hidden state
        self.encode_proj = nn.Linear(2 * hidden_size, hidden_size)

        # this is for projecting the tensor which is catenated with the attention and decoded hidden state to the hidden state needing to be processed
        self.decode_proj = nn.Linear(3 * hidden_size, hidden_size)

        self.proj2words = nn.Linear(hidden_size, len(tokenizer.tgt))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src: Tensor, src_lengths: List[int], tgt: Tensor):
        encoded_states, state = self.encode(src, src_lengths)
        enc_mask = self._generate_enc_mask(src, src_lengths)

        ps_log = self.decode(tgt, state, encoded_states, enc_mask)
        assert ps_log.shape == (tgt.size(0), tgt.size(1) - 1, len(self.tokenizer.tgt))
        # compute the loss
        loss = -torch.gather(ps_log, dim=-1, index=tgt[:, 1:].unsqueeze(dim=-1)).squeeze(dim=-1)
        tgt_mask = (tgt != self.tokenizer.tgt['<pad>']).float()
        loss *= tgt_mask[:, 1:]
        return loss.sum()

    def encode(self, x: Tensor, lengths: List[int]):
        """_summary_

        Args:
            x (Tensor): shape (batch_size, word_ids)
            lengths (List[int]): shape (batch_size, length)

        Returns:
            _type_: _description_
        """
        batch_size, L = x.shape
        x = self.embedding_layer.source_embed(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        packed_sequence, (h, c) = self.encoder(packed_x)
        all_states, _ = pad_packed_sequence(packed_sequence, batch_first=True)

        assert all_states.shape == (batch_size, L, 2 * self.hidden_size)
        assert h.shape == (
            2 * 1, batch_size, self.hidden_size), f"expected h.shape == (2, {batch_size}, {self.hidden_size}), but got ({h.shape}"

        h = torch.cat([*h], dim=1)
        c = torch.cat([*c], dim=1)
        return all_states, (self.get_initial_h_state(h), self.get_initial_c_state(c))

    def decode(self, x: Tensor, state: Tuple, encoded_states: Tensor, enc_mask: Tensor):
        x = x[:, :-1]  # Chop </s> token for max length sentences
        x = self.embedding_layer.target_embed(x)
        batch_size, seq_len, embed_size = x.shape

        o = torch.zeros(batch_size, self.hidden_size, device=self.device)
        projected_encoded_states = self.encode_proj(encoded_states)
        ps = []
        for t in range(seq_len):
            y_hat = torch.cat([o, x[:, t]], dim=1)
            state = self.decoder(y_hat, state)
            h, c = state
            attn = self._compute_attention(
                h, encoded_states, projected_encoded_states, enc_mask)
            v = self.decode_proj(torch.cat([attn, h], dim=1))
            o = self.dropout(F.tanh(v))
            p = F.log_softmax(self.proj2words(o), dim=1)
            ps.append(p)
        return torch.stack(ps, dim=1)

    def _compute_attention(self, h, encoded_states, projected_encoded_states, enc_mask=None):
        """ Compute the attention weight using dot product between decoded hidden state and the projected embedded hidden state.
        @param h (Tensor): tensor of shape (batch_size, hidden_size)
        @param encoded_states (Tensor): tensor of shape (batch_size, seq_len, 2 * hidden_size)
        @returns attn (Tensor): tensor of shape (batch_size, 2 * hidden_size)
        """
        attn_weight = torch.bmm(projected_encoded_states,
                                h.unsqueeze(2)).squeeze(2)
        if enc_mask is not None:
            attn_weight.masked_fill_(enc_mask == 0, -float('inf'))
        attn_weight = F.softmax(attn_weight, dim=1)
        attn = torch.bmm(attn_weight.unsqueeze(dim=1),
                         encoded_states).squeeze(dim=1)
        return attn

    def _generate_enc_mask(self, src: Tensor, src_lengths: List[int]):
        mask = torch.zeros_like(src)
        for i, src_length in enumerate(src_lengths):
            mask[i, :src_length] = 1
        return mask.to(self.device)

    def beam_search(self, src: Tensor, src_lengths: List[int], beam_size=5, max_decoding_time_step=70):
        """

        Args:
            src (Tensor): (1, L)
            src_lengths (List[int]): _description_
            beam_size (int, optional): _description_. Defaults to 5.
            max_decoding_time_step (int, optional): _description_. Defaults to 70.

        Returns:
            _type_: _description_
        """
        encoded_states, dec_state = self.encode(src, src_lengths)
        projected_encoded_states = self.encode_proj(encoded_states)

        # initial state: only one hypothesis
        hypotheses = [['<s>']]
        eos_id = self.tokenizer.tgt['</s>']
        hypo_scores = torch.zeros(len(hypotheses), device=self.device)

        # conserve the completed hypotheses
        completed_hypotheses = []

        o = torch.zeros(len(hypotheses), self.hidden_size, device=self.device)

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1

            last_words = torch.tensor([self.tokenizer.tgt[h[-1]] for h in hypotheses], dtype=torch.long, device=self.device)
            y = self.embedding_layer.target_embed(last_words)
            y_bar = torch.cat([y, o], dim=1)
            h, c  = self.decoder(y_bar, dec_state)
            assert h.size() == (len(hypotheses), self.hidden_size)

            encoded_states_expand = encoded_states.expand(len(hypotheses), -1, -1)
            projected_encoded_states_expand = projected_encoded_states.expand(len(hypotheses), -1, -1)
            attn = self._compute_attention(h, encoded_states=encoded_states_expand, projected_encoded_states=projected_encoded_states_expand)
            v = self.decode_proj(torch.cat([attn, h], dim=1))
            o = self.dropout(F.tanh(v))
            p = F.log_softmax(self.proj2words(o), dim=1)
            assert p.size() == (len(hypotheses), len(self.tokenizer.tgt))

            # update the score of each hypothesis
            scores_expand = hypo_scores.unsqueeze(dim=1).expand(-1, p.size(1))
            # print(f"p device: {p.device}, scores_expand: {scores_expand.device}")
            all_potential_hypotheses_scores = (scores_expand + p).view(-1)
            scores, indices = torch.topk(all_potential_hypotheses_scores, k=beam_size - len(completed_hypotheses))

            # update the hypotheses
            new_hypo = []
            new_hypo_scores = []
            new_hypo_indices = []
            for i, index in enumerate(indices):
                hypo_idx = index.item() // p.size(1)
                word_idx = index.item() % p.size(1)
                if word_idx == eos_id: 
                    completed_hypotheses.append(Hypothesis(value=hypotheses[hypo_idx][1:], score=scores[i].item()))
                else:
                    new_hypo.append(hypotheses[hypo_idx] + [self.tokenizer.tgt.id2word[word_idx]])
                    new_hypo_indices.append(hypo_idx)
                    new_hypo_scores.append(scores[i])
            hypo_scores = torch.tensor(new_hypo_scores, device=self.device)
            hypotheses = new_hypo

            # update the new dec_state
            new_h = h[torch.LongTensor(new_hypo_indices)]
            new_c = h[torch.LongTensor(new_hypo_indices)]
            dec_state = (new_h, new_c)
            o = o[torch.LongTensor(new_hypo_indices)]
            
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hypo_scores[0].item()))

        completed_hypotheses.sort(key=lambda hypo: hypo.score, reverse=True)
        return completed_hypotheses

    @property
    def device(self):
        return self.embedding_layer.source_embed.weight.device

    @staticmethod
    def load(model_path: str):
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = LstmNMT(tokenizer=params['vocab'], **args)
        model.load_state_dict(params['model'])
        print(f"load model from {model_path}")
        return model

    def save(self, path: str):
        print(f"save model to {path}")
        params = {
            'args': dict(embed_size=self.embedding_layer.embed_size,
                         hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.tokenizer,
            'model': self.state_dict()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(params, path)
