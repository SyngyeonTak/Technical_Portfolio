import random
from collections import Counter
import numpy as np

def make_learnable_dataset_sid(dataset):
    """
    dataset: list of (sequence, session_id, is_original) tuples
    returns: (inputs, targets, session_ids, is_original_flags)
    """
    input_sequences   = [seq[:-1] for seq, sid, is_original in dataset]
    target_items      = [seq[-1] for seq, sid, is_original in dataset]
    session_ids       = [sid for seq, sid, is_original in dataset]
    is_original_flags = [is_original for seq, sid, is_original in dataset]

    return (input_sequences, target_items, session_ids, is_original_flags)

def get_all_item_occurance(dataset):
    all_items = [item for sequence in dataset for item in sequence]

    # Count the occurrences of each item
    occurrences = Counter(all_items)

    # Display the occurrences
    return occurrences

def prefix_cropping_sid(original_dataset):
    """
    original_dataset: list of (sequence, session_id, is_original=True)
    returns: list of (cropped_sequence, session_id, is_original=False)
    """
    prefix_cropping_dataset = []

    for sequence, sid, _ in original_dataset:
        loop_range = len(sequence) - 2
        for idx in range(loop_range):
            modified_sequence = sequence[:-(idx + 1)]
            prefix_cropping_dataset.append((modified_sequence, sid, False))  # prefix → False

    return prefix_cropping_dataset


def meta_hybrid_scores(
    dataset,
    sim_mat,
    attn_log,
    alpha=0.5,
    window_size=5
):
    attn_dict = {idx: weights for idx, weights in attn_log}
    results = []

    half_w = window_size // 2

    for session, session_idx, _ in dataset:
        weights = attn_dict[session_idx]
        session = np.array(session)
        item_scores = []

        for i, (cur_id, w) in enumerate(zip(session[:-1], weights)):
            cur_index = cur_id - 1

            left = max(0, i - half_w)
            right = min(len(session), i + half_w + 1)
            neigh_idx = [session[j] - 1 for j in range(left, right) if j != i]
            context_score = sim_mat[cur_index, neigh_idx].mean()

            rep_score = w
            hybrid_score = alpha * context_score + (1 - alpha) * rep_score

            item_scores.append((cur_id, hybrid_score, i))

        results.append((session_idx, item_scores))

    return results

def binary_search_fdataset_hybrid_wo_rm(
    original_dataset, 
    target_ratio,        # 🎯 목표 비율
    sim_mat,
    attn_log,
    alpha=0.5,           # context vs rep_weight 비율
    k=1
):
    """
    Hybrid 방식 (context + rep_weight) 이진탐색 필터링.
    """
    augmented_dataset = original_dataset + prefix_cropping_sid(original_dataset)
    full_size = len(augmented_dataset)
    target_count = full_size - (full_size *target_ratio)

    score_set = meta_hybrid_scores(
        original_dataset,
        sim_mat=sim_mat,
        attn_log=attn_log,
        alpha=alpha,
    )

    score_set = make_extracted_sorted(score_set, k=k)

    fdataset_removed, removed_items = remove_items_from_dataset(original_dataset,score_set=score_set,target_count=target_count)

    faugmented_dataset = original_dataset + prefix_cropping_sid(fdataset_removed)

    return faugmented_dataset, removed_items

def make_extracted_sorted(fdataset, k=1):
    """
    fdataset: [(session_idx, [(item_id, hybrid_score, pos), ...]), ...]
    k: 각 세션에서 최소 hybrid_score 기준으로 뽑을 아이템 개수
    descending: True면 hybrid_score 기준 내림차순 정렬, False면 오름차순
    """
    extracted = []

    for session_idx, session_result in fdataset:
        sorted_items = sorted(session_result, key=lambda x: x[1])
        for _, score, pos in sorted_items[:k]:
            extracted.append((session_idx, score, pos))

    # hybrid_score 기준 정렬
    extracted_sorted = sorted(
        extracted, key=lambda x: x[1], reverse=False
    )

    return extracted_sorted

from collections import defaultdict
def remove_items_from_dataset(original_dataset, score_set, target_count):
    count = 0

    score_dict = defaultdict(list)
    for session_idx, score, pos in score_set:
        score_dict[session_idx].append([score, pos])
    # pos 기준 정렬 (혹은 hybrid_score 기준 정렬)
    for sid in score_dict:
        score_dict[sid].sort(key=lambda x: x[1])  # pos 순서로 정렬


    # 세션 인덱스별로 빠른 접근용 dict
    session_map = {sid: list(session) for session, sid, _ in original_dataset}
    removed_items = []
    count = 0
    for session_idx, score, pos in score_set:
        if count >= target_count:
            break

        session = session_map[session_idx]
        if len(session) <= 2:
            continue

        # 제거
        removed_item = session[pos]
        new_session = session[:pos] + session[pos+1:]
        if len(new_session) >= 2:
            session_map[session_idx] = new_session
            removed_items.append((session_idx, pos, removed_item, score))
            count += 1

            # 🔑 pos 업데이트: 해당 세션 dict에서만 처리
            for entry in score_dict[session_idx]:
                if entry[1] > pos:
                    entry[1] -= 1  # pos 앞으로 땡기기

    # 최종 dataset 재구성
    fdataset_removed = [
        (session_map[sid], sid, True)
        for _, sid, _ in original_dataset
        if sid in session_map
    ]

    return fdataset_removed, removed_items

def random_remove_items_from_dataset(original_dataset, target_count):
    # 세션 인덱스 리스트
    session_indices = list(range(len(original_dataset)))
    random.shuffle(session_indices)  # 세션 순서 랜덤화

    # 세션 복사본 (원본 보존)
    session_map = {i: list(original_dataset[i]) for i in session_indices}

    removed_items = []
    count = 0

    # 각 세션 한 번씩만 접근
    for sid in session_indices:
        if count >= target_count:
            break

        session = session_map[sid]
        if len(session) <= 2:
            continue  # 너무 짧은 세션은 패스

        # 무작위 아이템 제거
        pos = random.randint(0, len(session) - 1)
        removed_item = session[pos]

        # 제거 후 길이 확인
        new_session = session[:pos] + session[pos+1:]
        if len(new_session) >= 2:
            session_map[sid] = new_session
            removed_items.append((sid, pos, removed_item))
            count += 1

    # 최종 dataset 재구성
    fdataset_removed = [session_map[i] for i in sorted(session_map.keys())]

    return fdataset_removed, removed_items
