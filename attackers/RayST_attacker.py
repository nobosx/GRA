from attackers.basic_attacker import BasicAttacker
from victim_model import VictimModel
import torch
import numpy as np

class RaySTAttacker(BasicAttacker):
    """
    RayS-T attack.
    """
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
        assert (not self.targeted) or (target_y is not None), "An target_y (target label) must be provided in a targeted attack!"
        assert (not self.targeted) or (target_x is not None), "An target_x (auxiliary target image) must be provided in a targeted attack!"
        shape = list(x.shape)
        assert shape[0] == 1, "RayS Attack only supports batch size of 1 right now!"
        dim = np.prod(shape[1:])        

        # init variables
        self.queries = 0
        self.d_t = np.inf

        # The only changes compared with RayS Attack
        self.sgn_t = torch.sign(torch.ones(shape))
        if self.targeted:
            first_target_x = torch.unsqueeze(target_x[0], dim=0)
            self.sgn_t = torch.sign(first_target_x - x)
            self.sgn_t[torch.abs(self.sgn_t) < 0.5] = 1.0

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