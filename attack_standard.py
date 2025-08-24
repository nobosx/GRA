from attackers.load_attack import build_concrete_attacker
from victim_model import VictimModel
from data_sets import load_test_data
from evaluator import AttackProcess, ResultParser
from config import DATASET_NAME, MODEL_NAME, SUPPORTED_DATAMODELS, SUPPORTED_ATTACK_METHOD
from utils import build_record_path
import numpy as np
import argparse
import torch
import random

def parse_attack_config():
    parser = argparse.ArgumentParser(description="Standard Adversarial Attack")

    # Basic attack parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help=f"Dataset name. Supported datasets are {DATASET_NAME}.")
    parser.add_argument('--model', type=str, default='resnet50', help=f"Model name. Supported models are {MODEL_NAME}. Note that supported dataset-model combinations are {SUPPORTED_DATAMODELS}!")
    parser.add_argument('--method', type=str, default='RayS', help=f"Attack method. Supported attack methods are {SUPPORTED_ATTACK_METHOD}.")
    parser.add_argument('--norm', type=str, default='inf', choices=['inf', '2'], help="Distance norm, '2' or 'inf'")
    parser.add_argument('--targeted', type=int, default=1, help="1 for targeted attack and 0 for untargeted attack")
    parser.add_argument('--test_num', type=int, default=1000, help="Number of test images")
    parser.add_argument('--max_query_count', type=int, default=1000, help="Maximum query number limit")
    parser.add_argument('--pool_num', type=int, default=1, help="Adv image pool size")
    parser.add_argument('--test_batch_size', type=int, default=500, help="Batch size to test model accuracy")    
    parser.add_argument('--early_stop', type=int, default=0, help="1 for early stopping an attack")
    parser.add_argument('--epsilon', type=float, default=None, help="Epsilon metric for early stopping choice")
    parser.add_argument('--verbose', type=int, default=1, help="1 for printing process information")
    parser.add_argument('--test_model_acc', action='store_true', help="Set this param to test the accuracy of the victim model")
    parser.add_argument('--results_folder', type=str, default="./attack_records/", help=f"The folder where the resulsts are saved. Default is \'./attack_records/\'")

    # Special attack parameters
    # - For 'CGBA' and 'CGBA_H' attack
    parser.add_argument('--CGBA_dim_reduc_factor', type=int, default=1)
    # - For 'GuidedRay' attack
    parser.add_argument('--augment_method', type=str, default='gaussian_noise')

    args = parser.parse_args()
    attack_config = vars(args)
    attack_config['norm'] = np.inf if attack_config['norm'] == 'inf' else 2
    attack_config['targeted'] = attack_config['targeted'] == 1
    attack_config['early_stop'] = attack_config['early_stop'] == 1
    attack_config['verbose'] = attack_config['verbose'] == 1
    return attack_config

if __name__ == "__main__":
    # Set random seed
    rand_seed = 233
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    torch.backends.cudnn.deterministic = True

    attack_config = parse_attack_config()

    # Load test set
    attackset_loader = load_test_data(attack_config['dataset'], batch_size=1, shuffle=True)
    
    # Load victim model
    victim_model = VictimModel(attack_config['model'], attack_config['dataset'])

    # Load adversarial attacker
    attacker = build_concrete_attacker(victim_model, attack_config)

    # Set attack process
    attack_process = AttackProcess(victim_model, attackset_loader, attacker, attack_config['targeted'], attack_config['max_query_count'], test_batch_size=attack_config['test_batch_size'], target_pool_num=attack_config['pool_num'], verbose=attack_config['verbose'])
    
    # Test victim model accuracy
    if attack_config['test_model_acc']:
        accuracy = attack_process.test_victim_model_accuracy()
        print(f"Victim model accuracy: {accuracy}")

    # Perform attack
    attack_record = attack_process.mount_attack(attack_config['test_num'])

    # Refine attack record
    print("Refining attack record...")
    attack_record = ResultParser.refine_attack_record(attack_record)

    # Save attack record
    AttackProcess.save_attack_record(attack_record, build_record_path(attack_config))

    # Get ASR if an epsilon metric is provided
    if attack_config['epsilon'] is not None:
        ASR_value = ResultParser.parse_ASR_with_epsilons_and_queries(attack_record, [attack_config['epsilon']], [attack_config['max_query_count']])[0][0]
        print(f"Attack success rate is {ASR_value}.")