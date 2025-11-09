from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.index import index2ptr
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.data import HeteroData

EPS = 1e-15

class MetaPath2Vec(torch.nn.Module):
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,

        sparse: bool = False,
    ):
        super().__init__()

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        self.rowptr_dict, self.col_dict, self.rowcount_dict = {}, {}, {}

        self.rowptr_dict, self.col_dict, self.rowcount_dict = {}, {}, {}

        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = sort_edge_index(edge_index, num_nodes=max(sizes)).cpu()            
            rowptr = index2ptr(row, size=sizes[0])

            self.rowptr_dict[keys] = rowptr
            self.col_dict[keys] = col
            self.rowcount_dict[keys] = rowptr[1:] - rowptr[:-1]

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError(
                    "Found invalid metapath. Ensure that the destination node "
                    "type matches with the source node type across all "
                    "consecutive edge types.")

        assert walk_length + 1 >= context_size
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "The 'walk_length' is longer than the given 'metapath', but "
                "the 'metapath' does not denote a cycle")

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        types = {x[0] for x in metapath} | {x[-1] for x in metapath}
        types = sorted(list(types))

        count = 0 
        self.start, self.end = {}, {} 
        for key in types:
            self.start[key] = count 
            count += num_nodes_dict[key]
            self.end[key] = count

        self.item_embedding = Embedding(num_nodes_dict["item"], embedding_dim, sparse=sparse)

        self.dummy_idx = count

        self.reset_parameters()

    def reset_parameters(self):
        self.item_embedding.reset_parameters()

    def item_forward(self, batch: OptTensor = None) -> Tensor:
        emb = self.item_embedding.weight
        return emb if batch is None else emb.index_select(0, batch)
    
    def loader(self, **kwargs):
        return DataLoader(range(1, self.num_nodes_dict[self.metapath[0][0]]),
                          collate_fn=self._sample, **kwargs)
    
    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]

        item_index = [0]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            
            if edge_type[2] == 'item':
                item_index.append(i+1)

            batch = sample(
                self.rowptr_dict[edge_type],
                self.col_dict[edge_type],
                self.rowcount_dict[edge_type],
                batch,
                num_neighbors=1,
                dummy_idx=self.dummy_idx,
            ).view(-1)

            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw = rw[:, item_index]  # "item" 노드만 유지
        rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + len(item_index) - self.context_size
        
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]

        item_index = [0]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]

            if keys[2] == 'item':
                item_index.append(i+1)

            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        walks = []

        num_walks_per_rw = 1 + len(item_index) - self.context_size
        rw = rw[:, item_index]  # "item" 노드만 유지

        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    def item_loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.item_embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.item_embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.item_embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.item_embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, train_z: Tensor, train_y: Tensor, test_z: Tensor,
             test_y: Tensor, solver: str = "lbfgs", *args, **kwargs) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.embedding.weight.size(0) - 1}, '
                f'{self.embedding.weight.size(1)})')


def sample(rowptr: Tensor, col: Tensor, rowcount: Tensor, subset: Tensor,
           num_neighbors: int, dummy_idx: int) -> Tensor:

    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]

    rand = torch.rand((subset.size(0), num_neighbors), device=subset.device)
    rand *= count.to(rand.dtype).view(-1, 1)
    rand = rand.to(torch.long) + rowptr[subset].view(-1, 1)
    rand = rand.clamp(max=col.numel() - 1)  # If last node is isolated.

    col = col[rand] if col.numel() > 0 else rand
    col[mask | (count == 0)] = dummy_idx
    return col

def create_heterogeneous_data_neigh(session_data, category_dict, window_size = 5):
    """
    ✅ NetworkX 없이 바로 PyG의 HeteroData를 생성하는 함수
    """
    hetero_data = HeteroData()

    # ✅ 3. 엣지 저장을 위한 리스트 생성
    session_item_edges = []
    item_session_edges = []
    item_category_edges = []
    category_item_edges = []

    ############# window session 
    window_session_data = []
    window_session_id = 1

    for session in session_data:
        session_length = len(session)
        for i in range(session_length):

            left = max(0, i - (window_size // 2))
            right = min(session_length, i + (window_size // 2) + 1)

            in_item_id = session[i]
            out_item_ids = session[left:i] + session[i+1:right]

            item_session_edges.append([in_item_id, window_session_id])

            for out_item_id in out_item_ids:
                session_item_edges.append([window_session_id, out_item_id])
            
            window_session_data.append(session[left:right])
            window_session_id += 1

    # ✅ 1. 노드 ID 목록 생성
    item_set = set(item for session in session_data for item in session)
    category_set = set(category_dict.values())
    session_set = set(range(1, len(window_session_data) + 1))  # 세션 ID는 1부터 시작

    # ✅ 2. PyG에 노드 추가 (각 타입별로 따로 추가)
    hetero_data["item"].num_nodes = len(item_set)
    hetero_data["category"].num_nodes = len(category_set)
    hetero_data["session"].num_nodes = len(session_set)    

    # ✅ 5. 아이템-카테고리 엣지 추가
    for item, category in category_dict.items():
        item_category_edges.append([item, category])  # item → category
        category_item_edges.append([category, item])  # category → item

    # ✅ 6. 엣지를 PyTorch 텐서로 변환 후 HeteroData에 추가
    hetero_data["session", "contains", "item"].edge_index = torch.tensor(session_item_edges, dtype=torch.long).t()
    hetero_data["item", "belongs_to", "session"].edge_index = torch.tensor(item_session_edges, dtype=torch.long).t()
    hetero_data["item", "has_category", "category"].edge_index = torch.tensor(item_category_edges, dtype=torch.long).t()
    hetero_data["category", "contains_item", "item"].edge_index = torch.tensor(category_item_edges, dtype=torch.long).t()

    return hetero_data

def create_heterogeneous_data_all(session_data, category_dict):
    """
    ✅ NetworkX 없이 바로 PyG의 HeteroData를 생성하는 함수
    """
    hetero_data = HeteroData()
    

    # ✅ 1. 노드 ID 목록 생성
    item_set = set(item for session in session_data for item in session)
    category_set = set(category_dict.values())
    session_set = set(range(1, len(session_data) + 1))  # 세션 ID는 1부터 시작

    # ✅ 2. PyG에 노드 추가 (각 타입별로 따로 추가)
    hetero_data["item"].num_nodes = len(item_set)
    hetero_data["category"].num_nodes = len(category_set)
    hetero_data["session"].num_nodes = len(session_set)

    # ✅ 3. 엣지 저장을 위한 리스트 생성
    session_item_edges = []
    item_session_edges = []
    item_category_edges = []
    category_item_edges = []
    
    # ✅ 4. 세션-아이템 엣지 추가
    for session_id, items in enumerate(session_data, start=1):
        for item in items:
            session_item_edges.append([session_id, item])  # session → item
            item_session_edges.append([item, session_id])  # item → session

    # ✅ 5. 아이템-카테고리 엣지 추가
    for item, category in category_dict.items():
        item_category_edges.append([item, category])  # item → category
        category_item_edges.append([category, item])  # category → item

    # ✅ 6. 엣지를 PyTorch 텐서로 변환 후 HeteroData에 추가
    hetero_data["session", "contains", "item"].edge_index = torch.tensor(session_item_edges, dtype=torch.long).t()
    hetero_data["item", "belongs_to", "session"].edge_index = torch.tensor(item_session_edges, dtype=torch.long).t()
    hetero_data["item", "has_category", "category"].edge_index = torch.tensor(item_category_edges, dtype=torch.long).t()
    hetero_data["category", "contains_item", "item"].edge_index = torch.tensor(category_item_edges, dtype=torch.long).t()

    return hetero_data

def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in data_loader:
        optimizer.zero_grad()
        
        # ✅ DataParallel 처리
        target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        loss = target_model.item_loss(pos_rw.to(device), neg_rw.to(device))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def get_item_embeddings(model, device):
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    emb = target_model.item_forward().to(device)
    return emb.detach().cpu().numpy()
"""
if __name__ == "__main__":

    session_data = [
        [1, 2, 3, 4, 5, 6, 7, 8],  # Session 1
        [3, 4, 5, 9, 10, 11, 12, 13, 14],  # Session 2
        [6, 7, 8, 14, 15, 16, 17, 18],  # Session 3
        [19, 20, 17, 18, 21, 22, 23, 24, 25],  # Session 4
        [2, 3, 11, 14, 17, 23, 26, 27, 28]  # Session 5
    ]

    category_dict = {
        1: 1,  2: 2,  3: 3,  4: 1,  5: 2,  6: 3,  7: 1,  8: 2,
        9: 3, 10: 1, 11: 2, 12: 3, 13: 1, 14: 2, 15: 3, 16: 1,
        17: 2, 18: 3, 19: 1, 20: 2, 21: 3, 22: 1, 23: 2, 24: 3,
        25: 1, 26: 2, 27: 3, 28: 1
    }

    hetero_data = create_heterogeneous_data_all(session_data, category_dict)
    #hetero_data = create_heterogeneous_data_neigh(session_data, category_dict)

    #  MetaPath2Vec 설정
    metapath = [
        ("item", "belongs_to", "session"),
        ("session", "contains", "item"),
        ("item", "has_category", "category"),
        ("category", "contains_item", "item"),
        ("item", "belongs_to", "session"),
        ("session", "contains", "item")
    ]

    model = MetaPath2Vec(
        edge_index_dict=hetero_data.edge_index_dict,
        embedding_dim=16,
        metapath=metapath,
        walk_length=14,
        context_size=5,
        walks_per_node=1,
        num_negative_samples=2
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    data_loader = model.loader(batch_size=4, shuffle=True)

    epochs = 10  # 원하는 에폭 수 설정 가능
    for epoch in tqdm(range(epochs), desc='Epoch Progress'):
        loss = train(model, data_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    item_embeddings = get_item_embeddings(model, device)

    item_embeddings = item_embeddings[1:]

    item_sim = sim.calculate_cosine_similarity(item_embeddings)

    #print(len(item_embeddings))
"""