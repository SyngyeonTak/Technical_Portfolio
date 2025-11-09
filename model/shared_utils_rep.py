import torch
import numpy as np
import random    
import augment
import json
import os
import sys
from sklearn.decomposition import PCA

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_arguments_from_config(parser, model_name):
    config_path = os.path.join(f'./model/{model_name}/configs', f'{model_name}.json')
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found for model '{model_name}'.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        arg_configs = json.load(f)

    type_map = {'str': str, 'int': int, 'float': float, 'bool': bool}

    for arg_config in arg_configs:
        name = arg_config.pop('name')
        if 'type' in arg_config:
            arg_config['type'] = type_map.get(arg_config['type'])
        parser.add_argument(name, **arg_config)

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
    
def get_n_node(dataset_name):
    if dataset_name == 'Tmall':
        n_node = 40727 + 1

    return n_node

def get_train_loader(model_name, train_data, n_node = None, args = None, shuffle = False):
    if model_name == 'NARM':
        from NARM.dataset_rep import RecSysDataset
        from NARM.utils_rep import collate_fn
        from torch.utils.data import DataLoader
        
        train_data = RecSysDataset(train_data)
        train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = shuffle, collate_fn = collate_fn)
        return train_loader
    
    elif model_name == 'SR_GNN':
        from SR_GNN.pytorch_code.utils_rep import Data
        return Data(train_data, shuffle=shuffle)

    
    elif model_name == 'SHARE':
        from SHARE.utils_rep import Data
        train_loader = Data(train_data, args.window)
        return train_loader

    elif model_name == 'GNN_AM':
        from GNN_AM.utils_rep import Data
        return Data(train_data, shuffle=shuffle)
    
    else:
        raise NotImplementedError(f"'{model_name}': Train loader for model is not implemented.")


def get_test_loader(model_name, test_data, n_node = None, args = None, shuffle = False):
    if model_name == 'NARM':
        from NARM.dataset_rep import RecSysDataset
        from NARM.utils_rep import collate_fn
        from torch.utils.data import DataLoader
        
        test_data = RecSysDataset(test_data)
        test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = shuffle, collate_fn = collate_fn)
        return test_loader
    
    elif model_name == 'SR_GNN':
        from SR_GNN.pytorch_code.utils_rep import Data
        return Data(test_data, shuffle=shuffle)
    
    elif model_name == 'SHARE':
        from SHARE.utils_rep import Data
        test_loader = Data(test_data, args.window)
        return test_loader
    
    elif model_name == 'GNN_AM':
        from GNN_AM.utils_rep import Data
        return Data(test_data, shuffle=shuffle)

    else:
        raise NotImplementedError(f"'{model_name}': Test loader for model is not implemented.")

def get_model(model_name, args, n_node, original_dataset = None):
    if model_name == 'NARM':
        from NARM.narm_rep import NARM
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NARM(n_node, args).to(device)
        return model

    elif model_name == 'SR_GNN':
        from SR_GNN.pytorch_code.model_rep import SessionGraph
        model = trans_to_cuda(SessionGraph(args, n_node))
        return model

    elif model_name == 'GNN_AM':
        from GNN_AM.model_rep import SessionGraph
        model = trans_to_cuda(SessionGraph(args, n_node))
        return model
    
    elif model_name == 'SHARE':
        from SHARE.model_rep import SessionGraph
        model = trans_to_cuda(SessionGraph(args, n_node))
        return model

    else:
        raise NotImplementedError(f"'{model_name}': Model  is not implemented.")
    
def get_train_test_results(model_name, model, train_loader, test_loader, group_dict, seed):
    if model_name == 'NARM':
        from NARM.narm_rep import train_test_ht_sl
    elif model_name == 'SR_GNN':
        from SR_GNN.pytorch_code.model_rep import train_test_ht_sl
    
    elif model_name == 'SHARE':    
        from SHARE.model_rep import train_test_ht_sl    
    elif model_name == 'GNN_AM':    
        from GNN_AM.model_rep import train_test_ht_sl

    else:
        raise NotImplementedError(f"'{model_name}': train_test is not implemented.")
    
    return train_test_ht_sl(model, train_loader, test_loader, group_dict, seed)

def get_pretrained_embedding(model, pretrained_emb, emb_attr="embedding", freeze=False, pad_idx=0):

    import torch
    import numpy as np

    emb_layer = getattr(model, emb_attr)
    pretrained_emb = torch.tensor(pretrained_emb, dtype=torch.float32)
    pretrained_emb = pretrained_emb / np.linalg.norm(pretrained_emb, axis=1, keepdims=True)
    
    with torch.no_grad():
        emb_layer.weight[pad_idx+1:].copy_(pretrained_emb)  # 1~n
        emb_layer.weight[pad_idx].zero_()                  # 패딩은 zero

    emb_layer.weight.requires_grad = not freeze

    return model



