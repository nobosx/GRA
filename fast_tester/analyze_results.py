import argparse
import numpy as np
import os
from config import DATASET_NAME, MODEL_NAME, SUPPORTED_DATAMODELS, SUPPORTED_ATTACK_METHOD, SUPPORTED_DEFENSE_MODEL
from utils import build_record_path, build_record_path_defensive, build_ASR_DAT_path, build_ASR_DAT_path_defensive
from evaluator import AttackProcess, ResultParser

analysis_type_list = ['print_ASR', 'print_early_stop_query', 'write_ASR_with_queries']

def print_table_by_columns(columns: list, interval_space: str = " "):
    column_width = [max([len(i) for i in column]) for column in columns]
    row_num = len(columns[0])
    column_num = len(columns)
    for row_index in range(row_num):
        for col_index in range(column_num):
            format_str = "{" + f":^{column_width[col_index]}" + "}"
            print(format_str.format(columns[col_index][row_index]), end = interval_space if col_index != column_num - 1 else "\n")


def print_ASR(ASR_values: np.ndarray, epsilons: list, query_limit: list):
    interval_space = " " * 4

    # Readable format
    first_column = [""] + [f"epsilon={epsilon}" for epsilon in epsilons]
    ASR_columns = []
    for i in range(len(query_limit)):
        one_column = [f"query={query_limit[i]}"]
        for j in range(len(epsilons)):
            one_column.append(f"{ASR_values[j][i]:.4f}")
        ASR_columns.append(one_column)
    all_columns = [first_column] + ASR_columns
    print_table_by_columns(all_columns, interval_space)

    # LaTeX format
    # first_column = [""] + [f"{epsilon}" for epsilon in epsilons]
    # ASR_columns = []
    # for i in range(len(query_limit)):
    #     seq_column = ["&" for _ in range(len(epsilons) + 1)]
    #     ASR_columns.append(seq_column)
    #     one_column = [f"{query_limit[i]}"]
    #     for j in range(len(epsilons)):
    #         one_column.append(f"{ASR_values[j][i]:.3f}")
    #     ASR_columns.append(one_column)
    # end_columns = ["\\\\" for _ in range(len(epsilons) + 1)]
    # all_columns = [first_column] + ASR_columns + [end_columns]
    # print_table_by_columns(all_columns, interval_space)

def parse_attack_config():
    parser = argparse.ArgumentParser(description="Standard Adversarial Attack")

    # Basic attack parameters
    parser.add_argument('analysis_type', type=str, help=f"What type of analysis to do. Supported analysis type are {analysis_type_list}")
    parser.add_argument('--dataset', type=str, default='cifar10', help=f"Dataset name. Supported datasets are {DATASET_NAME}.")
    parser.add_argument('--model', type=str, default='resnet50', help=f"Model name. Supported models are {MODEL_NAME}. Note that supported dataset-model combinations are {SUPPORTED_DATAMODELS}!")
    parser.add_argument('--method', nargs='+', type=str, default=['RayS'], help=f"Attack method. Supported attack methods are {SUPPORTED_ATTACK_METHOD}.")
    parser.add_argument('--norm', type=str, default='inf', choices=['inf', '2'], help="Distance norm, '2' or 'inf'")
    parser.add_argument('--targeted', type=int, default=1, help="1 for targeted attack and 0 for untargeted attack")
    parser.add_argument('--test_num', type=int, default=1000, help="Number of test images")
    parser.add_argument('--max_query_count', type=int, default=1000, help="Maximum query number limit")
    parser.add_argument('--pool_num', type=int, default=5, help="Adv image pool size")
    parser.add_argument('--defense_model', type=str, default=None, help=f"Set this param if you want to show the results of a defense model. Supported defense models are {SUPPORTED_DEFENSE_MODEL}.")
    parser.add_argument('--results_folder', type=str, default="./attack_records/", help=f"The folder where the resulsts are saved. Default is \'./attack_records/\'")

    # Special attack parameters
    # - For 'GuidedRay' attack
    parser.add_argument('--augment_method', nargs='+', type=str)
    # - For 'power_bounce' attack
    parser.add_argument('--T', type=int, default=1)

    # Print ASR
    parser.add_argument('--query_limit', nargs='+', type=int)
    parser.add_argument('--epsilons', nargs='+', type=float)

    args = parser.parse_args()
    attack_config = vars(args)
    attack_config['norm'] = np.inf if attack_config['norm'] == 'inf' else 2
    attack_config['targeted'] = attack_config['targeted'] == 1
    return attack_config

def load_records_by_method(attack_config: dict):
    method_list = attack_config['method']
    record_by_method = dict()
    for method in method_list:
        attack_config['method'] = method
        if method == 'GuidedRay' and attack_config['targeted']:
            assert attack_config['augment_method'] is not None, "augment_method must be provided for targeted GuidedRay attack!"
            augment_method_list = attack_config['augment_method']
            for augment_method in augment_method_list:
                attack_config['augment_method'] = augment_method
                record_path = build_record_path(attack_config) if attack_config['defense_model'] is None else build_record_path_defensive(attack_config)
                record_by_method[f"{method}_{augment_method}"] = AttackProcess.load_attack_record(record_path)
            attack_config['augment_method'] = augment_method_list
        elif method == 'power_bounce':
            assert len(attack_config['augment_method']) == 1
            attack_config['augment_method'] = attack_config['augment_method'][0]
            record_path = build_record_path(attack_config) if attack_config['defense_model'] is None else build_record_path_defensive(attack_config)
            record_by_method[method] = AttackProcess.load_attack_record(record_path)
        else:
            record_path = build_record_path(attack_config) if attack_config['defense_model'] is None else build_record_path_defensive(attack_config)
            record_by_method[method] = AttackProcess.load_attack_record(record_path)
    return record_by_method

def write_ASR_data(ASR_values, x_values, save_path: str):
    assert len(ASR_values) == len(x_values)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode="w") as file:
        for i in range(len(x_values)):
            file.write(f"{x_values[i]} {ASR_values[i]}\n")

if __name__ == "__main__":
    attack_config = parse_attack_config()
    print("Parsed arguments are:")
    print(attack_config)

    # Check attack_config
    assert (attack_config['defense_model'] is None) or (attack_config['defense_model'] in SUPPORTED_DEFENSE_MODEL), f"Defense model {attack_config['defense_model']} is not supported!"
    assert attack_config['analysis_type'] in analysis_type_list, f"Parameter analysis_type must be in {analysis_type_list}! {attack_config['analysis_type']} was found!"

    if attack_config['analysis_type'] == 'print_ASR':
        # Print attack sucess rate
        assert (attack_config['query_limit'] is not None) and (attack_config['epsilons'] is not None), "query_limit and epsilon must be provided when printing ASR!"
        record_by_method = load_records_by_method(attack_config)
        for method in record_by_method.keys():
            attack_record = record_by_method[method]
            title = f"************************************Attack Success Rate for {method} Attack************************************"
            print(title)
            if method == 'power_bounce':
                for t in range(1 + attack_config['T']):
                    print(f"When T = {t}:")
                    ASR_values = ResultParser.parse_multi_ASR_with_epsilons_and_queries(attack_record, attack_config['epsilons'], attack_config['query_limit'], t)
                    print_ASR(ASR_values, attack_config['epsilons'], attack_config['query_limit'])
                    print('*' * len(title) + '\n')
            else:
                ASR_values = ResultParser.parse_ASR_with_epsilons_and_queries(attack_record, attack_config['epsilons'], attack_config['query_limit'])
                print_ASR(ASR_values, attack_config['epsilons'], attack_config['query_limit'])
                print('*' * len(title) + '\n')      
    elif attack_config['analysis_type'] == 'print_early_stop_query':
        # Print early stop query statistic
        assert (attack_config['query_limit'] is not None) and (len(attack_config['query_limit']) == 1), "query_limit must be provided and its length must be 1 when printing early stop query statistic!"
        assert (attack_config['epsilons'] is not None) and (len(attack_config['epsilons']) == 1), "epsilons must be provided and its length must be 1 when printing early stop query statistic!"
        record_by_method = load_records_by_method(attack_config)
        for method in record_by_method.keys():
            attack_record = record_by_method[method]
            early_stop_query_counts = ResultParser.parse_early_stop_query_count(attack_record, attack_config['epsilons'][0], attack_config['query_limit'][0])
            title = f"******************Early stop queries for {method} Attack with epsilon = {attack_config['epsilons'][0]} and query limit = {attack_config['query_limit'][0]}******************"
            print(title)
            print(f"ASR is {len(early_stop_query_counts) / attack_config['test_num']:.4f}")
            if len(early_stop_query_counts) == 0:
                print("No early stop queries found.")
            else:
                print(f"Early stop query statistics: mean = {np.mean(early_stop_query_counts):.4f}, median = {np.median(early_stop_query_counts):.4f}")
            print('*' * len(title) + '\n')
    elif attack_config['analysis_type'] == 'write_ASR_with_queries':
        # Write ASR values with queires to a .dat file
        assert attack_config['epsilons'] is not None, "An epsilon must be provided when parsing ASR!"
        record_by_method = load_records_by_method(attack_config)
        interval_num = 50
        assert attack_config['max_query_count'] % interval_num == 0
        interval_len = attack_config['max_query_count'] // interval_num
        queries = list(range(0, attack_config['max_query_count'] + interval_len, interval_len))
        epsilon_metric = attack_config['epsilons'][0]
        for method in record_by_method.keys():
            attack_record = record_by_method[method]
            ASR_values = ResultParser.parse_ASR_with_epsilons_and_queries(attack_record, [epsilon_metric], queries)[0]
            write_ASR_data(ASR_values, queries, build_ASR_DAT_path(attack_config, method, epsilon_metric) if attack_config['defense_model'] is None else build_ASR_DAT_path_defensive(attack_config, method, epsilon_metric))
        