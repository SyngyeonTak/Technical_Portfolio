import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from . import utils_rep
from tqdm import tqdm
import time

class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int): 
        n_layers(int): the number of gru layers

    """
    def __init__(self, n_items, args, n_layers = 1):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.n_layers = n_layers
        self.embedding_dim = args.embed_dim
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        #self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.parameters(), args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    def forward(self, seq, lengths, session_ids, is_original_flags, return_alpha=False):
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths, enforce_sorted=False)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        
        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)
        if return_alpha:
            sid_attn_pairs =  []
            masked_alpha = alpha.squeeze(-1).clone()  # (batch, seq_len)
            masked_alpha[mask.squeeze(-1) == 0] = float('-inf')

            # 마스크된 위치 제외하고 softmax
            softmax_alpha = torch.softmax(masked_alpha, dim=1)

            alpha_np = softmax_alpha.detach().cpu().numpy()
            mask_np = mask.squeeze(-1).cpu().numpy()

            for sid, attn_row, m_row, flag in zip(session_ids, alpha_np, mask_np, is_original_flags):
                if flag:
                    true_len = int(m_row.sum())
                    #sid_int = int(sid.item()) if torch.is_tensor(sid) else int(sid)
                    sid_attn_pairs.append((sid, attn_row[:true_len]))

            return scores, sid_attn_pairs
        else:
            return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)
        
def train_test_ht_sl(model, train_loader, test_loader, group_dict, seed):
    import numpy as np
    utils_rep.set_random_seed(seed)
    import datetime

    start_time = datetime.datetime.now()
    print('start training: ', start_time)

    model.train()

    sum_epoch_loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    attn_log = []  # ✅ original 세션의 attention weight 로그 저장
    
    for i, (seq, target, lens, session_ids, is_original_flags) in enumerate(train_loader):
        seq = seq.to(device)
        target = target.to(device)
        
        model.optimizer.zero_grad()
        #outputs = model(seq, lens)

        outputs, sid_attn_pairs = model(seq, lens, session_ids, is_original_flags, return_alpha = True)
        attn_log.extend(sid_attn_pairs)

        loss = model.criterion(outputs, target)
        loss.backward()
        model.optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

    print('\tLoss:\t%.3f' % sum_epoch_loss)
    end_time = datetime.datetime.now()
    print('start predicting: ', end_time)

    training_time = (end_time - start_time).total_seconds()
    model.eval()

    hit, mrr = [], []

    # head / tail
    head_hit, head_mrr = [], []
    tail_hit, tail_mrr = [], []

    evaluation_log = []
    score_log = []

    with torch.no_grad():
        for input_sessions, targets, lens, session_ids, is_original_flags in test_loader:
            input_sessions = input_sessions.to(device)
            
            targets = targets.to(device)
            # 모델 예측
            outputs = model(input_sessions, lens, session_ids, is_original_flags)          # [batch, n_items]
            logits = F.softmax(outputs, dim=1)             # 확률화
            sub_scores = torch.topk(logits, 20, dim=1).indices.cpu().numpy()
            targets = targets.cpu().numpy()
            eval_input_sessions = input_sessions.transpose(0, 1)
            eval_input_sessions = eval_input_sessions.cpu().numpy()
            # 배치 단위 평가
            for score, target, session, session_id in zip(sub_scores, targets, eval_input_sessions, session_ids):
                true_idx = target
                h = np.isin(true_idx, score)

                # mrr
                if true_idx in score:
                    m = 1 / (np.where(score == true_idx)[0][0] + 1)
                else:
                    m = 0

                item_type = ''
                # 전체
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
               
                # 로그 저장
                evaluation_log.append((session_id, target, h, m, item_type))
                score_log.append(score)

    # 평균 계산
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