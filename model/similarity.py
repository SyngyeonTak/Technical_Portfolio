import torch
import torch.nn.functional as F

gpu_index = 0

def calculate_cosine_similarity(embeddings, batch_size=1000):
    embeddings = torch.tensor(embeddings)
    embeddings = F.normalize(embeddings)
    num_rows = embeddings.shape[0]

    cosine_sim = torch.zeros((num_rows, num_rows))

    for i in range(0, num_rows, batch_size):
        batch_embeddings = embeddings[i : i + batch_size]
        cosine_sim[i : i + batch_size] = torch.mm(batch_embeddings, embeddings.T)
    
    return cosine_sim.cpu().numpy()

def separate_head_tail_by_pareto(node_degree_info):
    sorted_nodes = sorted(node_degree_info.items(), key=lambda x: x[1], reverse=True)
    top_20_percent_count = int(0.2 * len(sorted_nodes))
    
    head_items = [node for node, _ in sorted_nodes[:top_20_percent_count]]
    tail_items = [node for node, _ in sorted_nodes[top_20_percent_count:]]

    return head_items, tail_items

def get_highest_similarity(similarity_pairs, nodes_list):
    highest_similarity = []

    for node in nodes_list:
        if str(node) in similarity_pairs:
            pairs = similarity_pairs[str(node)]
            if not pairs:
                highest_similarity.append(0)
            else:
                max_pair = max(pairs, key=lambda x: x[1])
                max_pair_similarity = max_pair[1]
                highest_similarity.append(max_pair_similarity)

    highest_similarity = sorted(highest_similarity, reverse=True)
    
    return highest_similarity