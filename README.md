# 기술 수행 항목별 코드 링크

## 1️. 데이터사이언스 공통
**평가 항목:** Python 기반 데이터 처리 언어 활용 능력  
**수행 내용:**  
- 전반적으로 Python 언어를 사용하였으며, 특히 pytorch 라이브러리를 이용하여 GPU/CPU 환경에서 임베딩 간 유사도를 효율적으로 계산할 수 있는 병렬 연산 기반의 코사인 유사도 함수를 구현하였다.
예를 들어, 아래의 함수는 대규모 임베딩 행렬에 대해 배치 단위로 연산을 수행하여 메모리 사용량을 최소화하면서도 GPU 가속을 활용할 수 있도록 설계되었다.

**관련 코드:**  
- [similarity.py](similarity.py)
```python
import torch
import torch.nn.functional as F

def calculate_cosine_similarity(embeddings, batch_size=1000):
    embeddings = torch.tensor(embeddings)
    embeddings = F.normalize(embeddings)
    num_rows = embeddings.shape[0]

    cosine_sim = torch.zeros((num_rows, num_rows))

    for i in range(0, num_rows, batch_size):
        batch_embeddings = embeddings[i : i + batch_size]
        cosine_sim[i : i + batch_size] = torch.mm(batch_embeddings, embeddings.T)
    
    sim_min = cosine_sim.min()
    sim_max = cosine_sim.max()
    sim_mat_minmax = (cosine_sim - sim_min) / (sim_max - sim_min)
    sim_mat_minmax_cpu = sim_mat_minmax.cpu().numpy()
    
    return sim_mat_minmax_cpu
```
---

## 2️. 데이터 활용 및 분석
**평가 항목:** 머신러닝 라이브러리를 이용한 재현 가능한 개발 결과물 공개 여부  
**수행 내용:**  
- 기존 pytorch_geometirc의 metapath2vec 라이브러리를 SBRs(세션 기반 추천 시스템) 도메인에 맞게 커스터마이징 
- 사용 라이브러리: pytorch_geometirc
  
**관련 코드:**  
- [preprocess/metapath2vec.py](./preprocess/metapath2vec.py)
```
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.index import index2ptr
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.data import HeteroData


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
    rw = rw[:, item_index]
    rw[rw > self.dummy_idx] = self.dummy_idx

    walks = []
    num_walks_per_rw = 1 + len(item_index) - self.context_size
    
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + self.context_size])
    return torch.cat(walks, dim=0)

```
---

## 3️. 데이터 시각화
**평가 항목:** Python, R 등을 활용한 데이터 시각화 능력  
**수행 내용:**  
- `matplotlib`, `seaborn`을 이용하여 epoch별 성능 지표 시각화  

**관련 코드:**  
- [visualization_metrics.py](./visualization_metrics.py)  
- [plot_training_results.ipynb](./plot_training_results.ipynb)

---

📄 각 항목별 상세 설명은 상위 폴더 내 `README_detail.md` 참조
