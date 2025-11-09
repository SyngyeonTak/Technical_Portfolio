import argparse
import itertools
import os
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
current_file_path = __file__
current_file_name = os.path.splitext(os.path.basename(current_file_path))[0]
print(f"File name without extension: {current_file_name}")

from shared_utils_rep import (
    add_arguments_from_config,
    set_random_seed,
    get_n_node,
    get_train_loader,
    get_test_loader,
    get_model,
    get_train_test_results,
    get_pretrained_embedding,
    
)

from augment import (
    make_learnable_dataset_sid,
    get_all_item_occurance,
    binary_search_fdataset_hybrid_wo_rm
)

from similarity import (
    separate_head_tail_by_pareto,
    calculate_cosine_similarity,
)

def add_experimental_arguments(parser):
    parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
    parser.add_argument('--seed_list', nargs='+', default= [0], help='List of dataset names')
    parser.add_argument('--dataset_names', nargs='+', default= ['Tmall'], help='List of dataset names')
    parser.add_argument('--target_ratios', nargs='+', default= [0.98, 0.96, 0.94, 0.92, 0.90, 0.88], help='List of dataset names')
    parser.add_argument('--narm_epoch', type=int, default=100, help='the number of epochs to train for NARM')
    parser.add_argument('--sim_alphas', nargs='+', default= [0.50], help='List of dataset names')
    parser.add_argument('--emb_types', nargs='+', default= ['metapath2vec'], help='List of dataset names')
    parser.add_argument('--k_list', nargs='+', default= [1], help='List of dataset names')
    parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--model_names', nargs='+', default=['SR_GNN', 'SHARE'], help='List of model names')
    pre_args, remaining_argv = pre_parser.parse_known_args()

    total_combinations = []

    for model_name in pre_args.model_names:
        parser = argparse.ArgumentParser(parents=[pre_parser])
        add_arguments_from_config(parser, model_name)
        add_experimental_arguments(parser)

        args = parser.parse_args(remaining_argv)

        combinations = list(itertools.product(
            [model_name],
            args.dataset_names,
            args.emb_types,
            args.seed_list,
            args.target_ratios,
            args.sim_alphas,
            args.k_list
        ))
        total_combinations.extend(combinations)

    for model_name, dataset_name, emb_type, seed, target_ratio, sim_alpha, k in tqdm(total_combinations, desc="Running full combinations"):
        parser = argparse.ArgumentParser(parents=[pre_parser])
        add_arguments_from_config(parser, model_name)
        add_experimental_arguments(parser)
        args = parser.parse_args(remaining_argv)
        args.model_name = model_name
        args.dataset_name = dataset_name
        args.emb_type = emb_type
        args.seed = seed

        set_random_seed(seed)
        
        n_node = get_n_node(dataset_name)
    
        ##################### eval set
        original_dataset_raw = pickle.load(open('./datasets/' + dataset_name + f'/category/all_train_seq.txt', 'rb'))
        original_dataset = [(seq, idx, True) for idx, seq in enumerate(original_dataset_raw)]

        test_data = pickle.load(open('./datasets/' + dataset_name + f'/category/test.txt', 'rb'))

        sessions, targets = test_data
        session_indices = [i for i in range(len(sessions))]
        test_data = (sessions, targets, session_indices)

        ###################### 모델 별 loader, model 로드
        test_loader = get_test_loader(model_name, test_data, n_node = n_node, args=args, shuffle = False)

        model = get_model(model_name, args, n_node, original_dataset)
        
        if emb_type != 'baseline':
            if model_name == 'GNN_AM':
                emb_dim = 256
            elif model_name == 'NARM':
                emb_dim = 50
            else:
                emb_dim = 100

            embeddings = np.load(f"./embeddings/{emb_type}/{dataset_name}/node_embeddings_all_{emb_dim}.npy")

            model = get_pretrained_embedding(
                model,
                pretrained_emb=embeddings,
                emb_attr="emb" if model_name == 'NARM' else "embedding",
                freeze=False,
                pad_idx=0
            )

        ###################### 저장 경로 설정
        target_dir = f"./results/{model_name}/{current_file_name}/{dataset_name}/{emb_type}/target_ratio_{target_ratio}/sim_alpha_{sim_alpha}/seed_{seed}"
        os.makedirs(target_dir, exist_ok=True)
        all_records = []
        matrix_file = os.path.join(target_dir, f'results.csv')

        ###################### head, tail 나누기
        node_frequency_info = get_all_item_occurance(original_dataset_raw)
        head_list, tail_list = separate_head_tail_by_pareto(node_frequency_info)
        group_dict = {item: 'h' if item in head_list else 't' for item in head_list + tail_list}
    

        attn_log = [(idx, np.zeros(len(seq) - 1, dtype=np.float32)) for seq, idx, _ in original_dataset]

        emb_target_dir = f"./embeddings/sbrs/{model_name}/{dataset_name}"
        os.makedirs(emb_target_dir, exist_ok=True)

        best_result = [0, 0]
        bad_counter = 0

        train_epoch = args.narm_epoch if model_name == 'NARM' else args.epoch

        # -------------------------------
        #  학습 루프
        # -------------------------------

        for epoch in tqdm(range(train_epoch), desc="Training Progress"):
            print('-------------------------------------------------------')
            print('epoch:', epoch)
            
            sbrs_embedding = model.emb.weight if model_name == 'NARM' else model.embedding.weight
            sbrs_embedding = sbrs_embedding[1:]
            sbrs_embedding = sbrs_embedding.cpu().detach().numpy()

            sim_mat_minmax = calculate_cosine_similarity(sbrs_embedding, batch_size=100)
            
            alpha = 1.0 if epoch < 1 else sim_alpha

            train_data, removed_items = binary_search_fdataset_hybrid_wo_rm(
                original_dataset,
                target_ratio,
                sim_mat_minmax,
                attn_log,
                alpha=alpha,
                k=k
            )

            #train_data = train_data[:200]

            train_data = make_learnable_dataset_sid(train_data)
            train_loader = get_train_loader(model_name, train_data, n_node=n_node, args=args, shuffle = True)

            # --- 2. 학습 및 평가 ---
            results, _, _, raw_attn_log = get_train_test_results(
                model_name, model, train_loader, test_loader, group_dict, seed
            )
            attn_log = raw_attn_log

            # --- 로그 저장용 record ---
            row = {
                'epoch': epoch,
                'hit_all': results['overall_hit'],
                'hit_head': results['head_hit'],
                'hit_tail': results['tail_hit'],
                'mrr_all': results['overall_mrr'],
                'mrr_head': results['head_mrr'],
                'mrr_tail': results['tail_mrr'],
                'training_time': results['training_time'],
            }
            print(f"eval done")

            all_records.append(row)
            df_row = pd.DataFrame([row])
            if epoch == 0 and not os.path.exists(matrix_file):
                df_row.to_csv(matrix_file, index=False, mode='w')
            else:
                df_row.to_csv(matrix_file, index=False, mode='a', header=False)

            flag = 0
            if results['overall_hit'] > best_result[0]:
                best_result[0] = results['overall_hit']
                flag = 1
            if results['overall_mrr'] > best_result[1]:
                best_result[1] = results['overall_mrr']
                flag = 1

            bad_counter += 1 - flag
            if bad_counter >= args.patience:
                break

        print(f"\nCSV 저장 완료: {matrix_file} ({len(all_records)}개 레코드)")

if __name__ == '__main__':
    main()