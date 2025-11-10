# 기술 수행 항목별 코드 링크

## 1️. 데이터사이언스 공통
**평가 항목:** Python 기반 데이터 처리 언어 활용 능력  
**수행 내용:**  
- Python 기반으로 전체 파이프라인을 구현하였으며, `torch`, `torch.nn.functional` 모듈을 활용하여 **GPU/CPU 병렬 연산 환경에서 임베딩 간 코사인 유사도**를 효율적으로 계산하는 함수를 직접 설계함.  
- 입력 임베딩을 `torch.tensor()`로 변환 후 `F.normalize()`를 통해 L2 정규화하여, 내적(dot product)을 그대로 코사인 유사도로 활용할 수 있도록 구성함.  
- 전체 임베딩 행렬을 한 번에 연산하지 않고, `batch_size` 단위로 분할하여 `torch.mm()`을 수행함으로써 **메모리 사용량을 최소화**하고 대규모 임베딩 데이터셋에서도 OOM(Out Of Memory) 없이 동작하도록 함.  
- 계산된 유사도 행렬에 대해 `min–max scaling`을 적용하여 값을 [0, 1] 구간으로 정규화하고, 최종적으로 `.cpu().numpy()`를 통해 NumPy 배열로 변환하여 **GPU/CPU 환경 간 호환성**을 확보함.  
- NaN 또는 Inf 값 발생 시를 대비하여 예외 처리를 설계하였으며, `float16`, `bfloat16` 전환을 통해 **AMP(Automatic Mixed Precision)** 적용 시 추가적인 메모리 절약 및 연산 가속 가능.

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
- `torch_geometric`, `torch` 등의 라이브러리를 기반으로 **Metapath2Vec** 모델을 세션 기반 추천 시스템(SBRs) 도메인에 맞게 커스터마이징하여 재현함.  
- 기존 PyG 구현체에서 제공하는 `_pos_sample()` 함수를 수정하여, **item 노드 중심의 positive random walk sampling**을 수행하도록 구조를 개선함.  
- `metapath` 내의 엣지 타입을 순회하며 `sample()` 함수를 반복 호출하고, 각 단계에서 아이템 노드를 필터링하여 **session → item → session 구조를 반영한 메타패스 기반 임베딩**을 생성함.  
- `context_size`, `walk_length`, `walks_per_node` 등을 하이퍼파라미터로 받아 다양한 그래프 크기 및 메타 구조에 대응 가능하게 설계함.  
- 결과적으로 세션 내 아이템 간의 **의미적 인접성(semantic proximity)**을 유지하면서도, heterogeneous graph 구조를 효율적으로 학습할 수 있도록 최적화함.

**실행 방법:**  
- 최상 디렉터리에서 **python3 ./preprocess/main.py** 실행

**실행 결과물:**  
- [node_embeddings.npy](embeddings/metapath2vec/Tmall/node_embeddings_all_100.npy)

**관련 코드:**  
- [preprocess/metapath2vec.py](./preprocess/metapath2vec.py)
```python
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
- `matplotlib`, `seaborn` 라이브러리를 이용하여 학습 로그(`results.csv`)로부터 **epoch별 Hit@All 지표 변화**를 꺾은선 그래프로 시각화함.  
- `baseline`과 `metapath2vec` 두 방법을 동일 plot 내에서 `hue="method"`로 구분하여 비교 분석할 수 있도록 구현함.  
- 시각화 코드는 `Jupyter Notebook` 형태로 작성되어, 실험 재현 시 `results/SR_GNN/main/Tmall/.../results.csv` 파일만 경로 지정하면 즉시 실행 가능함.  
- 시각화 과정에서 `sns.lineplot()`에 `marker="o"`를 적용하여 각 epoch별 변화를 명확히 표현하고, `tight_layout()`을 사용하여 그래프의 여백 및 해상도를 최적화함.  
- 생성된 그래프는 학습 수렴 속도와 모델 안정성을 직관적으로 평가할 수 있는 근거 자료로 활용됨.
- 
**관련 코드:**  
- [visualize.ipynb](visualize.ipynb)  
---

