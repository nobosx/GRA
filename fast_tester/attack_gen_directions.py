from attackers.load_attack import build_concrete_attacker
from victim_model import VictimModel
from data_sets import load_test_data
from evaluator import AttackProcess, ResultParser
from config import DATASET_NAME, MODEL_NAME, SUPPORTED_DATAMODELS
from utils import build_record_path
import numpy as np
import argparse
import torch
import random

def build_attacker_info(victim_model, basic_config, attacker_configs):
    attacker_names = []
    attackers = []
    target_pool_nums = []
    total_configs = []
    for config in attacker_configs:
        config.update(basic_config)
        config['early_stop'] = False
        total_configs.append(config)
        attacker_names.append((config['method'] + f"_{config['augment_method']}") if config['method'] == 'GuidedRay' else config['method'])
        attackers.append(build_concrete_attacker(victim_model, config))
        target_pool_nums.append(config['pool_num'])
    return attacker_names, attackers, target_pool_nums, total_configs

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the Diversity of Generated Directions")
    parser.add_argument('--dataset', type=str, default='cifar10', help=f"Dataset name. Supported datasets are {DATASET_NAME}.")
    parser.add_argument('--model', type=str, default='resnet50', help=f"Model name. Supported models are {MODEL_NAME}. Note that supported dataset-model combinations are {SUPPORTED_DATAMODELS}!")
    parser.add_argument('--test_num', type=int, default=1000, help="Number of test images")
    parser.add_argument('--num_directions', type=int, default=10000, help="Number of generated directions for each image")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Basic configuration
    attack_config = {
        # Choose the dataset and model to be tested
        'dataset': args.dataset,
        'model': args.model,
        'norm': np.inf,
        'targeted': True,
        'test_num': args.test_num,
        'max_query_count': args.num_directions,
        'results_folder': './attack_records/gen_directions/',
        'reduction_method': 't-SNE',
        'verbose': True,
    }
    # Attackers to be tested
    attacker_configs = [
        {'method': 'RayS', 'pool_num': 1},
        {'method': 'GuidedRay', 'pool_num': 1, 'augment_method': 'gaussian_noise'},
        {'method': 'GuidedRay', 'pool_num': 1, 'augment_method': 'mix'},
        {'method': 'GuidedRay', 'pool_num': 5, 'augment_method': 'mix'},
    ]
    # Set random seed
    rand_seed = 233
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    torch.backends.cudnn.deterministic = True
    # Load test set
    attackset_loader = load_test_data(attack_config['dataset'], batch_size=1, shuffle=True)
    # Load victim model
    victim_model = VictimModel(attack_config['model'], attack_config['dataset'])
    # Build attacker information
    attacker_names, attackers, target_pool_nums, total_configs = build_attacker_info(victim_model, attack_config, attacker_configs)
    # Set attack process
    attack_process = AttackProcess(victim_model, attackset_loader, None, attack_config['targeted'], attack_config['max_query_count'], target_pool_num=target_pool_nums, verbose=attack_config['verbose'])
    # Test direction generation for attackers
    records = attack_process.test_init_directions(attack_config['test_num'], attackers, attacker_names)
    # Save each record in records
    for i, record in enumerate(records):
        record_path = build_record_path(total_configs[i])
        print(f"Saved generated directions for {attacker_names[i]} attacker pool num {total_configs[i]['pool_num']} to {record_path}")
        AttackProcess.save_attack_record(record, record_path)
    # Print the diversity statistics of generated directions for each attacker
    for i, record in enumerate(records):
        title = f"************************************Initialization test results for {attacker_names[i]} pool_num {target_pool_nums[i]} ************************************"
        print(title)
        print("Average cosine similarity of the initialization directions: {:.4f}".format(np.mean(record['cos_similarity'])))
        print('*' * len(title) + '\n')
    # Plot the data dimensionality reduction results of all methods
    record_by_method = {f"{attacker_names[i]} pool_num {target_pool_nums[i]}": records[i] for i in range(len(records))}
    ResultParser.plot_data_dimensionality_reduction(record_by_method, attack_config['reduction_method'], desc=f"{attack_config['dataset']} + {attack_config['model']}")