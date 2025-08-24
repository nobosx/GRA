import numpy as np

def build_record_path(attack_config: dict):
    if 'results_folder' not in attack_config:
        attack_config['results_folder'] = './attack_records/'
    if not attack_config['results_folder'].endswith('/'):
        attack_config['results_folder'] = attack_config['results_folder'] + '/'
    method = attack_config['method']
    if method == 'GuidedRay' and attack_config['targeted']:
        method = method + '_' + attack_config['augment_method']
    save_path = "{}dataset{}_model{}_{}_num{}_queries{}_norm{}_{}.pkl".format(attack_config['results_folder'], attack_config['dataset'], attack_config['model'], "targeted_pool{}".format(attack_config['pool_num']) if attack_config['targeted'] else "untargeted", attack_config['test_num'], attack_config['max_query_count'], attack_config['norm'], method)
    return save_path

def build_record_path_defensive(attack_config: dict):
    if 'results_folder' not in attack_config:
        attack_config['results_folder'] = './attack_records/'
    if not attack_config['results_folder'].endswith('/'):
        attack_config['results_folder'] = attack_config['results_folder'] + '/'
    method = attack_config['method']
    if method == 'GuidedRay' and attack_config['targeted']:
        method = method + '_' + attack_config['augment_method']
    save_path = "{}{}_dataset{}_model{}_{}_num{}_queries{}_norm{}_{}.pkl".format(attack_config['results_folder'], attack_config['defense_model'], attack_config['dataset'], attack_config['model'], "targeted_pool{}".format(attack_config['pool_num']) if attack_config['targeted'] else "untargeted", attack_config['test_num'], attack_config['max_query_count'], attack_config['norm'], method)
    return save_path

def build_ASR_DAT_path(attack_config: dict, method: str, epsilon_metric: float):
    if 'results_folder' not in attack_config:
        attack_config['results_folder'] = './attack_records/'
    if not attack_config['results_folder'].endswith('/'):
        attack_config['results_folder'] = attack_config['results_folder'] + '/'
    data_folder = attack_config['results_folder'] + 'ASR_values/'
    save_path = "{}ASR_epsilon{}_{}.dat".format(data_folder, epsilon_metric, method)
    return save_path

def build_ASR_DAT_path_defensive(attack_config: dict, method: str, epsilon_metric: float):
    if 'results_folder' not in attack_config:
        attack_config['results_folder'] = './attack_records/'
    if not attack_config['results_folder'].endswith('/'):
        attack_config['results_folder'] = attack_config['results_folder'] + '/'
    data_folder = attack_config['results_folder'] + 'ASR_values/'
    save_path = "{}{}_ASR_epsilon{}_{}.dat".format(data_folder, attack_config['defense_model'], epsilon_metric, method)
    return save_path

def cal_cos_similarity(data):
    """
    Calculate the average cosine similarity of the data.
    
    Args:
        data (np.ndarray): A 2D numpy array where each row is a data point.
        approx_num (int, optional): The number of data point pairs to approximate the cosine similarity. If None, use all data points.
    
    Returns:
        avg_cos_similarity float: The average cosine similarity.
    """
    if len(data) < 2:
        return 1.0
    # Flatten data if they are multi-dimensional
    data = np.reshape(data, (len(data), -1))
    # Calculate the average cosine similarity in a matrix way
    data_norms = np.linalg.norm(data, axis=1, ord=2)
    cos_similarity_matrix = np.dot(data, data.T) / (data_norms[:, None] * data_norms[None, :])
    avg_cos_similarity = np.mean(cos_similarity_matrix[np.triu_indices(len(data), k=1)])
    return avg_cos_similarity

def cal_mean_len(data):
    """
    Calculate the length of the mean vector of the data.

    Args:
        data (np.ndarray): A 2D numpy array where each row is a data point.

    Returns:
        mean_len (float): The length of the mean vector.
    """
    assert len(data) > 0, "Data should not be empty!"
    # Flatten data if they are multi-dimensional
    data = np.reshape(data, (len(data), -1))
    # Normalize each vector to unit length
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    mean_vector = np.mean(data, axis=0)
    mean_len = np.linalg.norm(mean_vector, ord=2)
    return mean_len
