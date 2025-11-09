import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm
from . import layers
from . import Modules
from . import utils_rep

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)



class HGNN_ATT(nn.Module):
    def __init__(self, dataset, input_size, n_hid, output_size, step, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.step = step
        self.dataset = dataset
        self.gat1 = layers.HyperGraphAttentionLayerSparse(input_size, n_hid, self.dropout, 0.2, transfer=False, concat=False)
        self.gat2 = layers.HyperGraphAttentionLayerSparse(n_hid, output_size, self.dropout, 0.2, transfer=True,  concat=False)
        
    def forward(self, x, H, G, EG):   

        residual = x

        x,y = self.gat1(x, H)

        if self.step == 2:

            x = F.dropout(x, self.dropout, training=self.training)
            x += residual
            x,y = self.gat2(x, H)

        x = F.dropout(x, self.dropout, training=self.training)
        x += residual

        return x, x



class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding2 = nn.Embedding(self.n_node, self.hidden_size)
        self.dropout = opt.dropout
        self.dataset = opt.dataset_name
        # for self-attention
        n_layers = 1
        n_head = 1
   
        
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.layer_stack = nn.ModuleList([
            Modules.EncoderLayer(self.hidden_size, self.hidden_size, n_head, self.hidden_size, self.hidden_size, dropout=opt.dropout)
            for _ in range(n_layers)])

        self.reset_parameters()
        

        self.hgnn = HGNN_ATT(self.dataset, self.hidden_size, self.hidden_size, self.hidden_size, opt.step, dropout = self.dropout)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, enc_output, enc_output2, mask, edge_mask, hidden, flags=None, session_ids=None, return_alpha=False):



        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask = get_pad_mask(mask, 0))
                    
        ht = enc_output[torch.arange(mask.shape[0]).long(), mask.shape[1]-1]  # batch_size x latent_size

        ht = self.layer_norm(ht)

        hidden = ht

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(hidden, b.transpose(1, 0))


        if return_alpha:
            sid_attn_pairs = []
            
            # 1. head 평균 → [batch, seq_len, seq_len]
            #attn_map = enc_slf_attn.mean(1).detach().clone()
            attn_map = enc_slf_attn.mean(dim=(1, 2)).detach().clone()

            # 2. 마스크 적용 (padding 위치 무시)
            #mask_2d = mask.unsqueeze(1).float()  # [batch, 1, seq_len]
            #attn_map = attn_map * mask_2d        # pad는 0으로

            attn_np = attn_map.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # 3. 각 세션별 attention row 저장
            for sid, attn_matrix, m_row, flag in zip(session_ids, attn_np, mask_np, flags):
                if flag:
                    true_len = int(m_row.sum())
                    truncated_attn = attn_matrix[-true_len:]
                    truncated_attn = torch.tensor(truncated_attn, device=enc_output.device, dtype=torch.float32)
                    normalized_attn = F.softmax(truncated_attn, dim=-1)
                    normalized_attn = normalized_attn.cpu().numpy()
                    sid_attn_pairs.append((sid, normalized_attn))

            return scores, sid_attn_pairs
        else:
            return scores



    def forward(self, inputs, HT, G, EG): 
        nodes = self.embedding(inputs) 
        #nodes = self.layer_norm1(nodes)       
        nodes, hidden = self.hgnn(nodes, HT, G, EG)
        nodes2 = self.embedding2(inputs) 
        return nodes,hidden,nodes2


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


def forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, session_ids_slice, is_original_slice, return_alpha=True):
    
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    HT = trans_to_cuda(torch.Tensor(HT).float())
    G = trans_to_cuda(torch.Tensor(G).float())
    EG = trans_to_cuda(torch.Tensor(EG).float())
    node_masks = trans_to_cuda(torch.Tensor(node_masks).long())
    edge_mask = trans_to_cuda(torch.Tensor(edge_mask).long())
    nodes, hidden, nodes2 = model(items, HT, G, EG)
    get = lambda i: nodes[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    get2 = lambda i: nodes2[i][alias_inputs[i]]
    seq_hidden2 = torch.stack([get2(i) for i in torch.arange(len(alias_inputs)).long()])

    if return_alpha:
        scores, sid_attn_pairs = model.compute_scores(
           seq_hidden, seq_hidden2, node_masks, edge_mask, hidden
           , flags=is_original_slice, session_ids=session_ids_slice, return_alpha=True
        )
        return targets, scores, sid_attn_pairs
    else:
        return targets, model.compute_scores(seq_hidden, seq_hidden2, node_masks, edge_mask, hidden)

def train_model(model, train_data, opt):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batchSize, True)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs = train_data.get_slice(i)    
        model.optimizer.zero_grad()
        targets, scores = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)

def test_model(model, test_data, opt):
    
    model.eval()
    hit20, mrr20, hit10, mrr10 = [], [], [], []
    slices = test_data.generate_batch(min(128,test_data.length), False)
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs = test_data.get_slice(i)
        targets, scores = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()

        for score, target in zip(sub_scores, targets):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1.0 / (np.where(score == target - 1)[0][0] + 1))

            hit10.append(np.isin(target - 1, score[:10]))
            if len(np.where(score[:10] == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1.0 / (np.where(score[:10] == target - 1)[0][0] + 1))
    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100
    return hit20, mrr20, hit10, mrr10

def train_test_ht_sl(model, train_data, test_data, group_dict, seed):
    utils_rep.set_random_seed(seed)
    import datetime
    model.scheduler.step()
    start_time = datetime.datetime.now()
    print('start training: ', start_time)
    

    model.scheduler.step()
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size, True)

    attn_log = []


    #for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
    for step in range(len(slices)):
        i = slices[step]
        alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, session_ids_slice, is_original_slice = train_data.get_slice(i)    
        model.optimizer.zero_grad()
        targets, scores, sid_attn_pairs = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs
                                                  , session_ids_slice, is_original_slice, return_alpha=True)
        
        attn_log.extend(sid_attn_pairs)
        
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)

    print('\tLoss:\t%.3f' % total_loss)
    end_time = datetime.datetime.now()
    print('start predicting: ', end_time)

    training_time = (end_time - start_time).total_seconds()


    model.eval()

    # 전체 결과
    hit, mrr = [], []

    # head / tail
    head_hit, head_mrr = [], []
    tail_hit, tail_mrr = [], []

    evaluation_log = []
    score_log = []

    
    slices = test_data.generate_batch(min(128,test_data.length), False)

    #for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
    for step in range(len(slices)):
        i = slices[step]
        alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, session_ids_slice, is_original_slice  = test_data.get_slice(i)
        targets, scores = forward(model, alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs
                                  , session_ids_slice, is_original_slice, return_alpha=False)
        
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        input_sessions = [test_data.inputs[idx] for idx in i]
        session_ids = [test_data.session_ids[idx] for idx in i]

        for score, target, session, session_ids in zip(sub_scores, targets, input_sessions, session_ids):
            true_idx = target - 1

            h = np.isin(true_idx, score)
            m = 1 / (np.where(score == true_idx)[0][0] + 1) if true_idx in score else 0
            item_type = ''

            # 전체
            hit.append(h)
            mrr.append(m)

            # head/tail 구분
            if group_dict.get(int(target.item())) == 'h':
                head_hit.append(h)
                head_mrr.append(m)
                item_type = 'h'
            else:
                tail_hit.append(h)
                tail_mrr.append(m)
                item_type = 't'

            evaluation_log.append((session_ids, target, h, m, item_type))  # ✅ 여기서 누적 저장
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
