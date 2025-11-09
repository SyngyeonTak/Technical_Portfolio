#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from . import utils_rep

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
    
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r) 
        inputgate = torch.sigmoid(i_i + h_i) 
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, flags=None, session_ids=None, return_alpha=False):
        ht = hidden[torch.arange(mask.shape[0]), torch.sum(mask, 1) - 1]

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (batch, seq_len, 1)

        # weighted sum (학습용, softmax 안 씀)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        b = self.embedding.weight[1:]
        scores = torch.matmul(a, b.transpose(1, 0))

        if return_alpha:
            sid_attn_pairs = []
            # 🔑 softmax 적용 (마스킹 포함)
            masked_alpha = alpha.squeeze(-1).clone()  # (batch, seq_len)
            masked_alpha[mask.squeeze(-1) == 0] = float('-inf')

            # 마스크된 위치 제외하고 softmax
            softmax_alpha = torch.softmax(masked_alpha, dim=1)

            alpha_np = softmax_alpha.detach().cpu().numpy()
            mask_np = mask.squeeze(-1).cpu().numpy()

            for sid, attn_row, m_row, flag in zip(session_ids, alpha_np, mask_np, flags):
                if flag:
                    true_len = m_row.sum()
                    #sid_int = int(sid.item()) if torch.is_tensor(sid) else int(sid)
                    sid_attn_pairs.append((sid, attn_row[:true_len]))

            return scores, sid_attn_pairs

        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, i, data, return_alpha=False):
    (alias_inputs, A, items, mask, targets,
     session_ids, is_original_flags, raw_inputs) = data.get_slice(i)

    alias_inputs = trans_to_cuda(torch.tensor(alias_inputs, dtype=torch.long))
    items = trans_to_cuda(torch.tensor(items, dtype=torch.long))
    mask = trans_to_cuda(torch.tensor(mask, dtype=torch.long))

    A_np = np.stack(A)
    A = trans_to_cuda(torch.tensor(A_np, dtype=torch.float))

    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    if return_alpha:
        scores, sid_attn_pairs = model.compute_scores(
            seq_hidden, mask, flags=is_original_flags, session_ids=session_ids, return_alpha=True
        )
        return targets, scores, sid_attn_pairs
    else:
        return targets, model.compute_scores(seq_hidden, mask)

def train_test_ht_sl(model, train_data, test_data, group_dict, seed):
    utils_rep.set_random_seed(seed)
    import datetime
    model.scheduler.step()
    start_time = datetime.datetime.now()
    print('start training: ', start_time)
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    attn_log = []  # ✅ original 세션의 attention weight 로그 저장

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, sid_attn_pairs = forward(model, i, train_data, return_alpha=True)

        attn_log.extend(sid_attn_pairs)
        
        targets = trans_to_cuda(torch.tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss

        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('\tLoss:\t%.3f' % total_loss)
    end_time = datetime.datetime.now()
    print('start predicting: ', end_time)

    training_time = (end_time - start_time).total_seconds()

    model.eval()

    hit, mrr = [], []
    head_hit, head_mrr = [], []
    tail_hit, tail_mrr = [], []

    slices = test_data.generate_batch(model.batch_size)

    evaluation_log = []
    score_log = []

    for batch_indices in slices:
        targets, scores = forward(model, batch_indices, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        input_sessions = [test_data.inputs[idx] for idx in batch_indices]
        session_ids = [test_data.session_ids[idx] for idx in batch_indices]

        for score, target, session, session_id in zip(sub_scores, targets, input_sessions, session_ids):
            true_idx = target - 1

            h = np.isin(true_idx, score)
            m = 1 / (np.where(score == true_idx)[0][0] + 1) if true_idx in score else 0
            item_type = ''

            hit.append(h)
            mrr.append(m)

            if group_dict.get(int(target.item())) == 'h':
                head_hit.append(h)
                head_mrr.append(m)
                item_type = 'h'
            else:
                tail_hit.append(h)
                tail_mrr.append(m)
                item_type = 't'

            evaluation_log.append((session_id, target, h, m, item_type))

            score_log.append(score)

    result = {
        'overall_hit': np.mean(hit) * 100,
        'overall_mrr': np.mean(mrr) * 100,
        'head_hit': np.mean(head_hit) * 100 if head_hit else 0,
        'head_mrr': np.mean(head_mrr) * 100 if head_mrr else 0,
        'tail_hit': np.mean(tail_hit) * 100 if tail_hit else 0,
        'tail_mrr': np.mean(tail_mrr) * 100 if tail_mrr else 0,
        'training_time': training_time    
    }

    return result, evaluation_log, score_log, attn_log

def train(model, train_data, seed):
    utils_rep.set_random_seed(seed)
    import datetime
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        try:
            targets, scores = forward(model, i, train_data)
        except Exception as e:
            print(f"🔥 Error at batch {i}: {e}")
            break

        targets = trans_to_cuda(torch.tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss

        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('\tLoss:\t%.3f' % total_loss)