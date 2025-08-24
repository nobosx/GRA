from attackers.basic_attacker import BasicAttacker
from victim_model import VictimModel
import torch
import numpy as np

class RaySAttacker(BasicAttacker):
    def __init__(self, model: VictimModel, norm_order: int, targeted: bool, max_query_count: int, early_stop: bool = False, epsilon_metric: float = None, verbose=True):
        super().__init__(model, norm_order, targeted, max_query_count, early_stop, epsilon_metric, verbose)
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        out = x + d * v
        return torch.clamp(out, lb, ub)
        
    def lin_search(self, x, y, target, sgn):
        d_end = np.inf
        for d in range(1, self.lin_search_rad + 1):
            if self.is_adversarial(self.get_xadv(x, sgn, d), y, target):
                d_end = d
                break
        return d_end
    
    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, target, sgn, tol=1e-3):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            if not self.is_adversarial(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                return False
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return False

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            if self.is_adversarial(self.get_xadv(x, sgn_unit, d_mid), y, target):
                d_end = d_mid
            else:
                d_start = d_mid
        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False
    

    def attack(self, x, y, target_x=None, target_y=None):
        assert target_x is None, "RayS Attack does not support a target_x input!"
        assert (not self.targeted) or (target_y is not None), "RayS Attack requires a target_y input for targeted attack!"
        shape = list(x.shape)
        assert shape[0] == 1, "RayS Attack only supports batch size of 1 right now!"
        dim = np.prod(shape[1:])

        # init variables
        self.queries = 0
        self.d_t = np.inf
        self.sgn_t = torch.sign(torch.ones(shape))
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
        dist = torch.tensor(np.inf)

        block_level = 0
        block_ind = 0

        attack_his = [(0, np.inf)]

        for i in range(self.max_query_count):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target_y, attempt)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0
            
            dist = torch.norm(self.x_final - x, self.norm_order)

            attack_his.append((self.queries, dist.item()))

            if self.queries >= self.max_query_count:
                break
            
            if self.early_stop and dist < self.epsilon_metric:
                print(f"Early stop at {self.queries} queries!")
                break
            
        return self.x_final, dist.item(), attack_his
    
    def check_valid_directions(self, x, y, target, sgn_attempts):
        ori_queries = self.queries
        for d in range(1, self.lin_search_rad + 1):
            try_flag = self.is_adversarial(self.get_xadv(x, sgn_attempts, d), y, target)
            if torch.sum(try_flag) > 0:
                self.queries = ori_queries
                for i in range(len(try_flag)):
                    if try_flag[i]:
                        valid_index = i
                        break
                self.queries += self.lin_search_rad * valid_index + d
                return valid_index
        return len(sgn_attempts)
    
    def test_initialization(self, x, y, target_x=None, target_y=None):
        """
        Test the initialization of the attack.

        Args:
            x (torch.Tensor): Input data. Each pixel value of x should be in the range of [0, 1].
            y (torch.Tensor): True labels for the input data.
            target_x (torch.Tensor, optional): Target data. Defaults to None.
            target_y (torch.Tensor, optional): Labels of target_x_batch. Defaults to None.

        Returns:
            hist (list[torch.Tensor]): Attack history in the initialization phase. Each element is a Ray direction with the same shape as x.

            query_count (int): Number of queries made during the initialization phase.

            success (bool): True if the initialization is successful, False otherwise.
        """
        assert target_x is None, "RayS Attack does not support a target_x input!"
        assert self.targeted and (target_y is not None), "Test initialization of RayS Attack only supports targeted attack with a target_y input!"
        shape = list(x.shape)
        assert shape[0] == 1, "RayS Attack only supports batch size of 1 right now!"
        dim = np.prod(shape[1:])

        if torch.min(x) < 0 or torch.max(x) > 1:
            print("Warning: Input x is not in [0, 1] range!")
            exit(0)

        # init variables
        self.queries = 0
        find_flag = False
        d_init = torch.sign(torch.ones(shape))
        self.sgn_t = torch.sign(torch.ones(shape))
        test_batch = 100
        hist = []

        block_level = 0
        block_ind = 0

        assert self.max_query_count % test_batch == 0
        while (self.queries < self.max_query_count) and (not find_flag):
            attempts = []
            for _ in range(test_batch):
                block_num = 2 ** block_level
                block_size = int(np.ceil(dim / block_num))
                start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

                attempt = d_init.clone().view(shape[0], dim)
                attempt[:, start:end] *= -1.
                attempt = attempt.view(shape)
                attempts.append(attempt)

                block_ind += 1
                if block_ind == 2 ** block_level or end == dim:
                    block_level += 1
                    block_ind = 0
            sgn_attempts = torch.concat(attempts, dim=0)
            valid_index = self.check_valid_directions(x, y, target_y, sgn_attempts)

            if valid_index < test_batch:
                find_flag = True
                hist += attempts[:valid_index + 1]
            else:
                hist += attempts

        return torch.concat(hist, dim=0).numpy(), self.queries, find_flag

    def gen_initialization_direction(self, x, y, target_x=None, target_y=None, num_directions=1000):
        """
        Generate a series of directions in the initialization phase.

        Args:
            x (torch.Tensor): Input data. Each pixel value of x should be in the range of [0, 1].
            y (torch.Tensor): True labels for the input data.
            target_x (torch.Tensor, optional): Target data. Defaults to None.
            target_y (torch.Tensor, optional): Labels of target_x_batch. Defaults to None.
            num_directions (int): The number of directions to be generated.

        Returns:
            directions (np.ndarray): The generated directions of size num_directions.
        """
        assert target_x is None, "RayS Attack does not support a target_x input!"
        assert self.targeted and (target_y is not None), "This method only supports targeted attack with a target_y input!"
        shape = list(x.shape)
        assert shape[0] == 1, "RayS Attack only supports batch size of 1 right now!"
        dim = np.prod(shape[1:])

        if torch.min(x) < 0 or torch.max(x) > 1:
            print("Warning: Input x is not in [0, 1] range!")
            exit(0)

        # init variables
        d_init = torch.sign(torch.ones(shape))
        directions = []

        block_level = 0
        block_ind = 0

        for _ in range(num_directions):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            attempt = d_init.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)
            directions.append(attempt)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

        return torch.concat(directions, dim=0).numpy()