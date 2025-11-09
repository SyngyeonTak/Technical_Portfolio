import argparse
import util
import os
import numpy as np
from tqdm import tqdm
import metapath2vec as mvc
import torch

def get_hetergeneous_mvc_item_embedding(args, num_epoch=5, embedding_dim=100):
    
    for dataset_name in args.datasets:

        print(dataset_name)
        
        category_path = f'datasets/{dataset_name}/category/category'

        dataset = util.get_dataset(dataset_name=dataset_name, if_category=True)
        
        ic_map = util.load_category(category_path)

        save_path = f'./embeddings/metapath2vec/{dataset_name}'

        hetero_data = mvc.create_heterogeneous_data_all(dataset, ic_map)

        metapath = [
            ("item", "belongs_to", "session"),
            ("session", "contains", "item"),
            ("item", "has_category", "category"),
            ("category", "contains_item", "item"),
            ("item", "belongs_to", "session"),
            ("session", "contains", "item")
        ]

        model = mvc.MetaPath2Vec(
            edge_index_dict=hetero_data.edge_index_dict,
            embedding_dim=embedding_dim,
            metapath=metapath,
            walk_length=100,
            context_size=5,
            walks_per_node=10,
            num_negative_samples=10
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loader_target = model.module if isinstance(model, torch.nn.DataParallel) else model
        data_loader = loader_target.loader(batch_size=100, shuffle=True)

        for epoch in tqdm(range(num_epoch), desc='Epoch Progress'):
            loss = mvc.train(model, data_loader, optimizer, device)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        item_embeddings = mvc.get_item_embeddings(target_model, device)
        item_embeddings = item_embeddings[1:] 
        print(f'item_embeddings length: {len(item_embeddings)}')

        os.makedirs(save_path, exist_ok=True)
        vectors_path = os.path.join(
            save_path, f"node_embeddings_all_{embedding_dim}.npy"
        )
        np.save(vectors_path, item_embeddings)

        print(f"Files saved in {save_path}")
        print('------------------------------------')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default= ['Tmall'],
                        help='List of dataset names')    
    args = parser.parse_args()

    get_hetergeneous_mvc_item_embedding(args, num_epoch=5, embedding_dim=100)

if __name__ == "__main__":
    main()

