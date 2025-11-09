
import torch
import pickle

gpu_index = 0
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

# Function to load the dataset from pickle file
def load_dataset(dataset_path):
    with open(f'{dataset_path}.txt', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def flatten_dataset(dataset):
    list_of_lists, targets = dataset
    
    if len(list_of_lists) != len(targets):
        raise ValueError("Length of list_of_lists and targets must be the same.")

    flattened_dataset = [lst + [target] for lst, target in zip(list_of_lists, targets)]
    
    return flattened_dataset

def get_dataset(dataset_name, length_type = 'all', if_augmented=False, if_category = False):
    base_path = f'./datasets/{dataset_name}/'

    if if_category:
        base_path += 'category/'

    if_flatten = False
    if length_type == 'all':
        dataset_path = f'{base_path}{"all_" if not if_augmented else ""}train{"_seq" if not if_augmented else ""}'

        if if_augmented:
                if_flatten = True
    
    else:
        dataset_path = f'{base_path}{length_type}_train{"_seq" if not if_augmented else ""}'

    dataset = load_dataset(dataset_path)

    if if_flatten:
        dataset = flatten_dataset(dataset)

    return dataset

def load_category(category_path):
    with open(f'{category_path}.txt', 'rb') as f:
        category = pickle.load(f)
    return category
