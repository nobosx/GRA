from victim_model import VictimModel
from attackers.basic_attacker import BasicAttacker
from attackers.RayS_attacker import RaySAttacker
from attackers import GuidedRay_attacker
from torch.utils.data import DataLoader
from data_sets import load_test_data
from config import CLASS_NUM
from time import time
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from utils import cal_cos_similarity, cal_mean_len
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import pickle
import os

class AttackProcess(object):
    """
    An object to execute the adversarial attack process.
    """

    def __init__(self, victim_model: VictimModel, attack_set_loader: DataLoader, attacker: BasicAttacker, targeted: bool, max_query_count: int, test_batch_size: int = 100, target_pool_num: int = None, verbose: bool = True):
        assert (not targeted) or (target_pool_num is not None), "Target pool number must be specified for targeted attacks!"
        self.victim_model = victim_model
        self.dataset_name = victim_model.dataset_name
        self.attack_set_loader = attack_set_loader
        self.attacker = attacker
        self.targeted = targeted
        self.max_query_count = max_query_count
        self.test_batch_size = test_batch_size
        self.target_pool_num = target_pool_num
        self.verbose = verbose
        if targeted:
            if self.dataset_name != "imagenet":
                self.ref_samples_by_class = self.__get_ref_samples_by_class(victim_model, load_test_data(self.dataset_name, self.test_batch_size, shuffle=False, attack_oriented=False), CLASS_NUM[self.dataset_name])
                if self.verbose:
                    for i in range(len(self.ref_samples_by_class)):
                        print(f"Class {i} has {len(self.ref_samples_by_class[i])} correctly classified samples.")
            else:
                # For imagenet dataset, we will get reference samples for one class later in the attack process
                with open("./data/imagenet/idx_to_class.pkl", "rb") as file:
                    self.idx_to_class = pickle.load(file)
                self.ref_samples_by_class = None

    def __get_ref_samples_by_class(self, model: VictimModel, data_loader: DataLoader, num_classes: int):
        ref_samples_by_class = [[] for _ in range(num_classes)]
        for x, y in data_loader:
            predict = model.query_hard_label(x)
            for i in range(len(x)):
                if predict[i] == y[i]:
                    ref_samples_by_class[y[i]].append(x[i])
        return ref_samples_by_class
    
    def __get_ref_samples_for_one_class(self, target_label):
        if self.ref_samples_by_class is not None:
            return self.ref_samples_by_class[target_label]
        else:
            assert self.dataset_name == "imagenet"
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            # Special process for imagenet
            target_dir_path = f"./data/imagenet/{self.idx_to_class[target_label]}/"
            ref_samples = []
            assert os.path.isdir(target_dir_path), f"Dir {target_dir_path} does not exist!"
            for fig_name in os.listdir(target_dir_path):
                if not fig_name.endswith('.JPEG'):
                    continue
                fig = Image.open(target_dir_path + fig_name).convert('RGB')
                fig = transform(fig)
                if self.victim_model.query_hard_label(torch.unsqueeze(fig, dim=0)).item() == target_label:
                    ref_samples.append(fig)
            assert len(ref_samples) >= 1
            return ref_samples
    
    def test_victim_model_accuracy(self):
        """
        Test the accuracy of the victim model on the test set.

        Returns:
            accuracy (float): The accuracy of the victim model on the test set.
        """
        testset_loader = load_test_data(self.dataset_name, self.test_batch_size, shuffle=True, attack_oriented=False)
        correct = 0
        total = 0
        if self.verbose:
            iterator = tqdm(testset_loader, desc="Testing model accuracy")
        else:
            iterator = testset_loader
        for x, y in iterator:
            predict = self.victim_model.query_hard_label(x)
            correct += torch.sum(predict == y).item()
            total += len(x)
        accuracy = correct / total
        return accuracy
    
    def mount_attack(self, attack_num: int = 10):
        """
        Mount an adversarial attack and output the results.

        Args:
            attack_num (int): The number of attacks to be tested.

        Returns:
            results (dict): A dictionary containing the results of the attack.
                - 'norm_type' (str): The norm type used for the attack, either 'inf' or an integer.
                - 'dist' (float): Distances between the original and adversarial examples.
                - 'hist' (list[list[(int, float)]]): Attack History. Each element is a tuple of (query_count, dist).
        """
        
        attack_record = {
            "norm_type": "inf" if self.attacker.norm_order == np.inf else int(self.attacker.norm_order),
            "distance": [],
            "hist": []
        }
        if self.verbose:
            print("#######################################################################################################################################################################################")
            print("Mounting {} attack against {} model on {} dataset under {} norm. Total test samples: {}.".format("targeted" if self.targeted else "untargeted", self.victim_model.model_name, self.dataset_name, attack_record["norm_type"], attack_num))
            print("#######################################################################################################################################################################################")

        test_count = 0
        for image, label in self.attack_set_loader:
            if test_count >= attack_num:
                break
            assert len(image) == 1, "Attack batch size must be 1 right now!"
            # Make sure the image is correctly classified by the victim model
            if self.victim_model.query_hard_label(image).item() != label.item():
                continue
            test_count += 1
            if self.verbose:
                print(f"Attacking test sample {test_count}/{attack_num}...")
            start_t = time()
            # Untargeted attack
            if not self.targeted:
                adv_x, dist, attack_his = self.attacker.attack(image, label)
            # Targeted attack
            else:
                # Randomly select a target class that is different from the original class
                target_y = torch.randint(0, CLASS_NUM[self.dataset_name], (1,))
                while target_y.item() == label.item():
                    target_y = torch.randint(0, CLASS_NUM[self.dataset_name], (1,))
                # Randomly select a target pool of reference samples from the same class
                target_x = random.choices(self.__get_ref_samples_for_one_class(target_y.item()), k=self.target_pool_num)
                target_x = torch.stack(target_x)
                if isinstance(self.attacker, RaySAttacker):
                    adv_x, dist, attack_his = self.attacker.attack(image, label, target_y=target_y)
                else:
                    adv_x, dist, attack_his = self.attacker.attack(image, label, target_x=target_x, target_y=target_y)
            end_t = time()

            # Add an initial record entry
            if attack_his[0][0] > 0:
                attack_his = [(0, np.inf)] + attack_his
            # Check query overflow
            # 1. Exclude overflow points
            overflow_allowed_margin = 1.1
            if attack_his[-1][0] > (self.max_query_count * overflow_allowed_margin):
                check_pos = len(attack_his) - 1
                while attack_his[check_pos][0] > (self.max_query_count * overflow_allowed_margin):
                    check_pos -= 1
                attack_his = attack_his[:check_pos+1]
            # 2. Margin the end
            check_pos = len(attack_his) - 1
            while attack_his[check_pos][0] > self.max_query_count:
                attack_his[check_pos] = (self.max_query_count, attack_his[check_pos][1])
                check_pos -= 1
            attack_record["distance"].append(dist)
            attack_record["hist"].append(attack_his)
            if self.verbose:
                print(f"Attacking time consumption: {end_t - start_t} s.")
                print(f"Final adversarial distance: {dist}.")
                print("")
        
        if isinstance(self.attacker, GuidedRay_attacker.GuidedRayAttacker):
            print(f"Init success rate is {GuidedRay_attacker.init_succ_count / test_count}.")
            print(f"Attack success rate is {GuidedRay_attacker.attack_succ_count / test_count}.")

        if self.verbose:
            if test_count < attack_num:
                print(f"Warning: Running out of qulified test samples. Only {test_count} samples were attacked out of {attack_num} requested.")
            print("#######################################################################################################################################################################################")
            print("All attacks completed!")
            print("#######################################################################################################################################################################################\n")
        return attack_record
    
    def test_attack_initialization(self, attack_num: int = 10):
        """
        Test the attack initialization process.

        Args:
            attack_num (int): The number of attacks to be tested.

        Returns:
            attack_record (dict): A dictionary containing the results of the attack initialization.
                - 'norm_type' (str): The norm type used for the attack, either 'inf' or an integer.
                - 'mean_len' (list[float]): The mean length of the initialization history.
                - 'cos_similarity' (list[float]): The average cosine similarity of the initialization history.
        """
        assert hasattr(self.attacker, "test_initialization"), "The attacker must have a test_initialization method!"
        assert self.targeted, "The attack must be targeted for testing initialization!"
        if self.verbose:
            print("#######################################################################################################################################################################################")
            print("Test initialization stage for {} attack against {} model on {} dataset under {} norm. Total test samples: {}.".format("targeted" if self.targeted else "untargeted", self.victim_model.model_name, self.dataset_name, self.attacker.norm_order, attack_num))
            print("#######################################################################################################################################################################################")
        test_count = 0
        attack_record = {
            "norm_type": "inf" if self.attacker.norm_order == np.inf else int(self.attacker.norm_order),
            "init_success": [],
            "init_queries": [],
        }

        for image, label in self.attack_set_loader:
            if test_count >= attack_num:
                break
            assert len(image) == 1, "Attack batch size must be 1 right now!"
            # Make sure the image is correctly classified by the victim model
            if self.victim_model.query_hard_label(image).item() != label.item():
                continue
            test_count += 1
            if self.verbose:
                print(f"Attacking test sample {test_count}/{attack_num}...")
            # Randomly select a target class that is different from the original class
            target_y = torch.randint(0, CLASS_NUM[self.dataset_name], (1,))
            while target_y.item() == label.item():
                target_y = torch.randint(0, CLASS_NUM[self.dataset_name], (1,))
            # Randomly select a target pool of reference samples from the same class
            target_x = random.choices(self.__get_ref_samples_for_one_class(target_y.item()), k=self.target_pool_num)
            target_x = torch.stack(target_x)
            if isinstance(self.attacker, RaySAttacker):
                hist, num_queries, success = self.attacker.test_initialization(image, label, target_y=target_y)
            else:
                hist, num_queries, success = self.attacker.test_initialization(image, label, target_x=target_x, target_y=target_y)
            attack_record["init_success"].append(success)
            attack_record["init_queries"].append(num_queries)
            if self.verbose:
                print(f"Success example: {False if attack_record['init_success_example'] is None else True}.")
                print(f"Initialization success: {success}.")
                print(f"Number of queries used in initialization: {num_queries}.")
                print("")

        if test_count < attack_num:
            print(f"Warning: Running out of qulified test samples. Only {test_count} samples were attacked out of {attack_num} requested.")
        print("#######################################################################################################################################################################################")
        print("All attacks completed!")
        print(f"Initialization success rate: {np.sum(attack_record['init_success']) / len(attack_record['init_success'])}.")
        print(f"Average number of queries used in initialization: {np.mean(attack_record['init_queries'])}.")
        print(f"Average cosine similarity of the initialization history: {np.mean(attack_record['cos_similarity'])}.")
        print(f"Average mean length of the initialization history: {np.mean(attack_record['mean_len'])}.")
        print("#######################################################################################################################################################################################\n")

        return attack_record
    
    def test_init_directions(self, attack_num: int = 10, attackers: list = None, attacker_names: list = None):
        """
        Test the initialization directions of the attack.

        Args:
            attack_num (int): The number of attacks to be tested.
            attackers (list): The list of attackers to be tested.
            attacker_names (list): The list of names of the attackers.
        """
        attack_records = []
        # Generate attacking data
        x_list = []
        y_list = []
        target_y_list = []
        test_count = 0
        for image, label in self.attack_set_loader:
            if test_count >= attack_num:
                break
            assert len(image) == 1, "Attack batch size must be 1 right now!"
            # Make sure the image is correctly classified by the victim model
            if self.victim_model.query_hard_label(image).item() != label.item():
                continue
            test_count += 1
            # Randomly select a target class that is different from the original class
            target_y = torch.randint(0, CLASS_NUM[self.dataset_name], (1,))
            while target_y.item() == label.item():
                target_y = torch.randint(0, CLASS_NUM[self.dataset_name], (1,))
            x_list.append(image)
            y_list.append(label)
            target_y_list.append(target_y)
        # Test each attacker
        for i in range(len(attackers)):
            if self.verbose:
                print("#######################################################################################################################################################################################")
                print(f"Testing initialization directions for {attacker_names[i]} attacker against {self.victim_model.model_name} model on {self.dataset_name} dataset with pool num {self.target_pool_num[i]}. Total attack num: {attack_num}.")
            attack_records.append({
                'mean_len': [],
                'cos_similarity': [],
                'init_success_example': None,
            })
            for j in range(attack_num):
                # Generate target_x for the attacker
                target_x = random.choices(self.__get_ref_samples_for_one_class(target_y.item()), k=self.target_pool_num[i])
                target_x = torch.stack(target_x)
                # Perform the initialization direction generation
                t1 = time()
                if isinstance(attackers[i], RaySAttacker):
                    directions = attackers[i].gen_initialization_direction(x_list[j], y_list[j], target_y=target_y_list[j], num_directions=self.max_query_count)
                else:
                    directions = attackers[i].gen_initialization_direction(x_list[j], y_list[j], target_x, target_y_list[j], self.max_query_count)
                t2 = time()
                mean_len = cal_mean_len(directions)
                avg_cos_similarity = cal_cos_similarity(directions)
                t3 = time()
                attack_records[i]['mean_len'].append(mean_len)
                attack_records[i]['cos_similarity'].append(avg_cos_similarity)
                if j == 0:
                    attack_records[i]['init_success_example'] = directions
                if self.verbose:
                    print(f"Test sample {j+1}/{attack_num} for {attacker_names[i]} attacker pool num {self.target_pool_num[i]}:")
                    print(f"Mean length of the initialization direction: {mean_len}.")
                    print(f"Average cosine similarity of the initialization direction: {avg_cos_similarity}.")
                    print(f"Time consumption for generating initialization direction: {t2 - t1} s. Time consumption for calculating statistics: {t3 - t2} s.")
                    print("")
            if self.verbose:
                print(f"Average mean length of the initialization direction for {attacker_names[i]} attacker pool num {self.target_pool_num[i]}: {np.mean(attack_records[i]['mean_len'])}.")
                print(f"Average cosine similarity of the initialization direction for {attacker_names[i]} attacker pool num {self.target_pool_num[i]}: {np.mean(attack_records[i]['cos_similarity'])}.")
                print("#######################################################################################################################################################################################\n")
        return attack_records
    
    @staticmethod
    def save_attack_record(attack_record, save_path):
        """
        Save the attack record to a file.

        Args:
            attack_record (dict): The attack record to be saved.
            save_path (str): The path where the attack record will be saved.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Use pickle to save the attack record
        with open(save_path, 'wb') as f:
            pickle.dump(attack_record, f)
    
    @staticmethod
    def load_attack_record(save_path):
        """
        Load the attack record from a file.

        Args:
            save_path (str): The path from where the attack record will be loaded.

        Returns:
            attack_record (dict): The loaded attack record.
        """
        with open(save_path, 'rb') as f:
            attack_record = pickle.load(f)
        return attack_record
    
class ResultParser(object):
    """
    An object to parse the attack results.
    """

    def __init__(self):
        pass

    @classmethod
    def __parse_ASR_with_queries(cls, dist_with_queries: list, epsilon_metric: float, max_queries: list):
        # Make sure that the max_queries is sorted in ascending order
        for i in range(len(max_queries) - 1):
            assert max_queries[i] < max_queries[i + 1], "max_queries must be sorted in ascending order!"
        test_num = len(dist_with_queries)
        test_pos = [0 for _ in range(test_num)]
        list_length = [len(dists) for dists in dist_with_queries]
        ASR_with_queries = []
        for query_restriction in max_queries:
            success_count = 0
            for i in range(test_num):
                while test_pos[i] < list_length[i] and dist_with_queries[i][test_pos[i]][0] <= query_restriction:
                    test_pos[i] += 1
                success_count += 1 if test_pos[i] > 0 and dist_with_queries[i][test_pos[i] - 1][1] < epsilon_metric else 0
            ASR_with_queries.append(success_count / test_num)
        return ASR_with_queries
    
    @classmethod
    def __parse_ASR_with_epsilons(cls, dist_with_queries: list, epsilons: list, query_restriction: int):
        epsilon_len = len(epsilons)
        success_num = [0 for _ in range(epsilon_len)]
        test_num = len(dist_with_queries)
        for i in range(test_num):
            pos = 0
            length = len(dist_with_queries[i])
            while pos < length and dist_with_queries[i][pos][0] <= query_restriction:
                pos += 1
            for j in range(epsilon_len):
                success_num[j] += 1 if pos > 0 and dist_with_queries[i][pos - 1][1] < epsilons[j] else 0
        return [i / test_num for i in success_num]      
    
    @classmethod
    def parse_ASR_with_epsilons_and_queries(cls, attack_record: dict, epsilons: list, max_queries: list):
        """
        Parse ASR values with epsilon metrics and query restrictions from an attack record.

        Args:
            attack_record (dict): The attack record to be parsed.
            epsilons (list[float]): The epsilon metrics at which ASR values are to be parsed.
            max_queries (list[int]): The query number restrictions at which ASR values are to be parsed. max_queries must be sorted in ascending order.

        Returns:
            ASR_values (np.ndarray): A 2d-ndarray with ASR_values[i][j] being the ASR value at epsilons[i] and max_queries[j].
        """    
        if isinstance(epsilons, float) or isinstance(epsilons, int):
            epsilons = [epsilons]
        if isinstance(max_queries, float) or isinstance(max_queries, int):
            max_queries = [max_queries]

        if len(epsilons) < len(max_queries):
            # Call ASR_with_queries
            ASR_values = []
            for epsilon_metric in epsilons:
                ASR_values.append(cls.__parse_ASR_with_queries(attack_record["hist"], epsilon_metric, max_queries))
            ASR_values = np.array(ASR_values)
        else:
            # Call ASR_with_epsilons
            ASR_values = []
            for query_restriction in max_queries:
                ASR_values.append(cls.__parse_ASR_with_epsilons(attack_record["hist"], epsilons, query_restriction))
            ASR_values = np.transpose(np.array(ASR_values))
        return ASR_values
    
    @classmethod
    def parse_distance_with_queries(cls, attack_record: dict, max_queries: list, statistic: str = 'median'):
        """
        Parse distance statistic with query_restrictions from an attack record.

        Args:
            attack_record (dict): The attack record to be parsed.
            max_queries (list[int]): The query number restrictions at which ASR values are to be parsed. max_queries must be sorted in ascending order.
            statistic (str): Statistic of distance. 'median' or 'mean'.

        Returns:
            distance_statistic_with_queries (np.ndarray): An 1d-ndarray with distance_statistic_with_queries[i] is the distance statistic with max query number being max_queries[i].
        """
        if isinstance(max_queries, int):
            max_queries = [max_queries]
        # Make sure that the max_queries is sorted in ascending order
        for i in range(len(max_queries) - 1):
            assert max_queries[i] < max_queries[i + 1], "max_queries must be sorted in ascending order!"
        assert statistic in ['median', 'mean'], f"Statistic type must be 'median' or 'mean'! {statistic} is not supported!"
        
        dist_with_queries = attack_record['hist']
        test_num = len(dist_with_queries)
        test_pos = [0 for _ in range(test_num)]
        distance_array = np.zeros((test_num,))
        list_length = [len(dists) for dists in dist_with_queries]
        distance_statistic_with_queries = []
        for query_restriction in max_queries:
            for i in range(test_num):
                while test_pos[i] < list_length[i] and dist_with_queries[i][test_pos[i]][0] <= query_restriction:
                    test_pos[i] += 1
                if test_pos[i] == 0:
                    distance_array[i] = np.inf
                else:
                    distance_array[i] = dist_with_queries[i][test_pos[i] - 1][1]
            distance_statistic_with_queries.append(np.median(distance_array) if statistic == 'median' else np.mean(distance_array))
        return np.array(distance_statistic_with_queries)
    
    @classmethod
    def parse_early_stop_query_count(cls, attack_record: dict, epsilon_metric: float, query_restriction: int):
        """
        Get the early stop query count from an attack record.

        Args:
            attack_record (dict): The attack record to be parsed.
            epsilon_metric (float): The epsilon metric for the early stop.
            query_restriction (int): The max query number restriction for the early stop.
        
        Returns:
            early_stop_query_count (np.ndarray): The early stop query count of each successful attack.
        """
        early_stop_query_counts = []
        attack_hist = attack_record["hist"]
        for dist_with_query_entry in attack_hist:
            # Find the first early stop position
            pos = 0
            while pos < len(dist_with_query_entry) and dist_with_query_entry[pos][1] > epsilon_metric:
                pos += 1
            if pos >= len(dist_with_query_entry):
                # Can not find the early stop position
                continue
            early_stop_query = dist_with_query_entry[pos][0]
            if early_stop_query <= query_restriction:
                early_stop_query_counts.append(early_stop_query)
        return np.array(early_stop_query_counts, dtype=np.int32)
    
    @classmethod
    def refine_attack_record(cls, attack_record: dict):
        """
        Refine the attack record by:
        1. making sure that each hist entry has a zero query count at the beginning,
        2. replacing np.inf distance with 1.0 for linf norm attacks,
        3. making sure that each hist entry has non-increasing distances with non-decreasing query counts,
        4. removing 'distance' entry from the attack record.

        Args:
            attack_record (dict): The attack record to be refined.

        Returns:
            refined_record (dict): The refined attack record.
        """
        refined_record = {
            "norm_type": attack_record["norm_type"],
        }
        attack_hist = deepcopy(attack_record["hist"])
        # Check norm type
        assert refined_record["norm_type"] in ["inf", "2"], f"Norm type must be 'inf' or '2'! {refined_record['norm_type']} is not supported!"
        num_entries = len(attack_hist)

        # 1. Make sure that each hist entry has a zero query count at the beginning
        for i in range(num_entries):
            if attack_hist[i][0][0] > 0:
                attack_hist[i] = [(0, np.inf)] + attack_hist[i]

        # 2. Replace np.inf distance with 1.0 for linf norm attacks
        if refined_record["norm_type"] == "inf":
            for i in range(num_entries):
                for j in range(len(attack_hist[i])):
                    if attack_hist[i][j][1] == np.inf:
                        attack_hist[i][j] = (attack_hist[i][j][0], 1.0)
        
        # 3. Make sure that each hist entry has non-increasing distances with non-decreasing query counts
        for i in range(num_entries):
            for j in range(1, len(attack_hist[i])):
                assert attack_hist[i][j][0] >= attack_hist[i][j - 1][0], f"Query counts must be non-decreasing! {attack_hist[i][j]} < {attack_hist[i][j - 1]}!"
                if attack_hist[i][j][1] > attack_hist[i][j - 1][1]:
                    print(f"Warning: entry {attack_hist[i][j]} has larger distance than entry {attack_hist[i][j - 1]}! This may be caused by the attack overflow. The distance will be replaced with the previous distance.")
                    attack_hist[i][j] = (attack_hist[i][j][0], attack_hist[i][j - 1][1])

        refined_record["hist"] = attack_hist

        return refined_record
    
    @classmethod
    def plot_data_dimensionality_reduction(cls, record_by_method: dict, reduction_method: str = "PCA", desc: str = "Cifar10 + ResNet50"):
        """
        Plot the data dimensionality reduction results of different methods.

        Args:
            record_by_method (dict): A dictionary where keys are method names and values are attack records.
            reduction_method (str): The dimensionality reduction method to be used, either 'PCA' or 't-SNE'.
        """
        assert reduction_method in ["PCA", "t-SNE"], f"Reduction method must be 'PCA' or 't-SNE'! {reduction_method} is not supported!"
        # Construct data concatenation
        all_hist_data = []
        labels = []
        method_to_label = {method_name: i for i, method_name in enumerate(record_by_method.keys())}
        for method_name, attack_record in record_by_method.items():
            attack_example = attack_record['init_success_example'] if attack_record['init_success_example'] is not None else attack_record['init_failure_example']
            print(f"Processing attack method {method_name} with {len(attack_example)} examples.")
            all_hist_data.append(attack_example)
            labels.append(np.ones(len(attack_example), dtype=np.int32) * method_to_label[method_name])
        all_hist_data = np.concatenate(all_hist_data, axis=0)
        labels = np.concatenate(labels, axis=0)
        # Flatten the data
        all_hist_data = all_hist_data.reshape(all_hist_data.shape[0], -1)
        # Standardize the data
        scaler = StandardScaler()
        all_hist_data = scaler.fit_transform(all_hist_data)
        # Perform dimensionality reduction
        if reduction_method == "PCA":
            reduced_data = PCA(n_components=2).fit_transform(all_hist_data)
        else:
            reduced_data = TSNE(n_components=2, metric='cosine').fit_transform(all_hist_data)
        # Plot the reduced data
        converted_method_names = {
            'RayS': 'RayS',
            'GuidedRay_gaussian_noise': 'GuidedRay-G',
            'GuidedRay_mix': 'GuidedRay-M',
        }
        color_list = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3']
        index = 0
        plt.figure(figsize=(20,16))
        for method_name, attack_record in record_by_method.items():
            method_label = method_to_label[method_name]
            method_data = reduced_data[labels == method_label]
            label_name = converted_method_names[method_name[:method_name.find(' ')]] + method_name[method_name.find(' '):].replace('pool_num ', 'k=')
            plt.scatter(method_data[:, 0], method_data[:, 1], c=color_list[index], label=label_name, alpha=0.4, s=30, marker='o')
            index += 1
        
        plt.title(f"{desc}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.show()