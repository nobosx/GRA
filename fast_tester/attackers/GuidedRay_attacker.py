from attackers.basic_attacker import BasicAttacker
from victim_model import VictimModel
from torchvision.transforms import RandomRotation, Compose, RandomCrop, Resize, ColorJitter
import torch
import numpy as np

SUPPORTED_IMAGE_AUGMENT_METHOD = ['gaussian_noise', 'rotation', 'crop', 'color_jitter', 'mix']

init_succ_count = 0
init_failed_count = 0
succ_for_init_success_count = 0
succ_for_init_failed_count = 0
attack_succ_count = 0

class GuidedRayAttacker(BasicAttacker):
    def __init__(self, model: VictimModel, norm_order: int, targeted: bool, max_query_count: int, early_stop: bool = False, epsilon_metric: float = None, verbose=True, augment_method='gaussian_noise'):
        super().__init__(model, norm_order, targeted, max_query_count, early_stop, epsilon_metric, verbose)
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None
        self.lin_search_rad = 10
        self.image_augment_method = augment_method
        global SUPPORTED_IMAGE_AUGMENT_METHOD
        assert augment_method in SUPPORTED_IMAGE_AUGMENT_METHOD, f"Image augment method {augment_method} is not supported!"

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
        if d_end <= self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False
    
    def image_augment(self, image, try_image_augment_method=None):
        global SUPPORTED_IMAGE_AUGMENT_METHOD
        if try_image_augment_method is None:
            image_augment_method = self.image_augment_method
        else:
            image_augment_method = try_image_augment_method
        assert image_augment_method in SUPPORTED_IMAGE_AUGMENT_METHOD, f"Image augumentation method {self.image_augment_method} is not supported!"
        if image_augment_method == 'gaussian_noise':
            # Add gaussian noise
            if image.shape[-1] == 32:
                perturb_d = 5.0
            elif image.shape[-1] == 224:
                perturb_d = 100.0
            rand_perturb = torch.randn_like(image)
            rand_perturb = rand_perturb / torch.norm(rand_perturb).item()
            new_image = self.get_xadv(image, rand_perturb, perturb_d)
        elif image_augment_method == 'rotation':
            # Rotate the image
            rotation_degree = 180
            transform = RandomRotation(rotation_degree)
            new_image = transform(image)
            assert new_image.shape == image.shape
        elif image_augment_method == 'crop':
            # Randomly crop the image
            assert image.shape[-1] == image.shape[-2]
            crop_ratio = 0.5
            croped_size = int(crop_ratio * image.shape[-1])
            transform = Compose([
                RandomCrop(croped_size),
                Resize((image.shape[-2], image.shape[-1]))
            ])
            new_image = transform(image)
        elif image_augment_method == 'color_jitter':
            # Change the brightness, contrast, saturation and hue
            change_ratio = 1.0
            transform = ColorJitter(brightness=change_ratio, contrast=change_ratio, saturation=change_ratio, hue=0.5 * change_ratio)
            new_image = transform(image)
            assert new_image.shape == image.shape
        elif image_augment_method == 'mix':
            augment_method_pool = ['gaussian_noise', 'rotation', 'crop', 'color_jitter']
            # Randomly choose an augment method
            new_image = self.image_augment(image, try_image_augment_method=augment_method_pool[np.random.randint(0, len(augment_method_pool))])
        return new_image


    def attack(self, x, y, target_x=None, target_y=None):
        global init_succ_count, init_failed_count, attack_succ_count, succ_for_init_failed_count, succ_for_init_success_count
        assert (not self.targeted) or (target_y is not None), "GuidedRay Attack requires a target_y input for targeted attack!"
        shape = list(x.shape)
        assert shape[0] == 1, "RayS Attack only supports batch size of 1 right now!"
        dim = np.prod(shape[1:])

        # init variables
        self.queries = 0

        # Initialization of the attack
        if self.targeted:
            # Targeted attack scenario
            max_query_count_to_find_direction = self.max_query_count
            test_batch = 100
            assert max_query_count_to_find_direction % test_batch == 0
            find_flag = False
            while (not find_flag) and self.queries < max_query_count_to_find_direction:
                try_x_adv = target_x[torch.randint(0, target_x.shape[0], (test_batch,))]
                new_x_adv = torch.stack([self.image_augment(x_adv) for x_adv in try_x_adv])
                direction_sgn = torch.sign(new_x_adv - x)
                # test norm2
                d = torch.norm(torch.reshape(new_x_adv - x, (test_batch, -1)), p=2, dim=1)
                d = torch.reshape(d, (test_batch, 1, 1, 1))
                try_flag = self.is_adversarial(self.get_xadv(x, direction_sgn / torch.norm(direction_sgn[0], p=2).item(), d), y, target_y)
                if torch.sum(try_flag) > 0:
                    # Find a good direction
                    find_flag = True
                    for i in range(test_batch):
                        if try_flag[i]:
                            self.queries -= test_batch - i - 1
                            direction_sgn = direction_sgn[i]
                            d = d[i].item()
                            break
        # Untargeted attack scenario
        else:
            find_flag = False

            # Plan1
            stride = 0.03
            max_stride = 0.03
            test_limit = 50
            for i in range(test_limit):
                direction_sgn = torch.sign(torch.randn_like(x))
                if stride * (i + 1) < max_stride:
                    d = stride * (i + 1)
                else:
                    d = max_stride
                new_x_adv = self.get_xadv(x, direction_sgn, d)
                d *= torch.norm(direction_sgn).item()
                try_flag = self.is_adversarial(new_x_adv, y).item()
                if try_flag:
                    find_flag = True
                    break

        if find_flag:
            init_succ_count += 1
            self.sgn_t = direction_sgn
            self.d_t = d
            self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
            self.binary_search(x, y, target_y, self.sgn_t)
            if self.verbose:
                print(f"Find a good direction with {self.queries} queries!")
        else:
            init_failed_count += 1
            self.d_t = np.inf
            self.sgn_t = torch.sign(torch.ones(shape))
            self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
            if self.verbose:
                print(f"Cannot find a good direction with {self.queries} queries!")

        block_level = 0
        block_ind = 0

        dist = torch.norm(self.x_final - x, p=self.norm_order)
        attack_his = [(self.queries, dist)]
        if self.queries >= self.max_query_count:
            # Running out of queries during initialization
            if self.verbose:
                print(f"Init success rate up to now is {init_succ_count / (init_succ_count + init_failed_count)}.")
            return self.x_final, dist.item(), attack_his

        for i in range(self.max_query_count):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            # attempt = torch.reshape(self.sgn_t.clone(), (shape[0], dim))
            attempt = self.sgn_t.clone().contiguous().view(shape[0], dim)
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
                attack_succ_count += 1
                if self.verbose:
                    print(f"Early stop at {self.queries} queries!")
                break
        
        if self.verbose:
            print(f"Init success rate up to now is {init_succ_count / (init_succ_count + init_failed_count)}.")

        # Debug
        if self.early_stop:
            if dist.item() < self.epsilon_metric:
                if find_flag:
                    succ_for_init_success_count += 1
                else:
                    succ_for_init_failed_count += 1
            if self.verbose:
                print(f"Success rate for init success case is {succ_for_init_success_count / init_succ_count if init_succ_count > 0 else 0.0}. Success rate for init failure case is {succ_for_init_failed_count / init_failed_count if init_failed_count > 0 else 0.0}.")
                print(f"Init success rate up to now is {init_succ_count / (init_succ_count + init_failed_count)}. Success rate up to now is {(succ_for_init_failed_count + succ_for_init_success_count) / (init_succ_count + init_failed_count)}.")
        
        return self.x_final, dist.item(), attack_his
    
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
        assert self.targeted, "Test initialization of GuidedRay Attack only supports targeted attack!"
        assert (target_x is not None) and (target_y is not None), "target_x and target_y inputs are required for testing initialization of GuidedRay Attack!"
        shape = list(x.shape)
        assert shape[0] == 1, "RayS Attack only supports batch size of 1 right now!"
        dim = np.prod(shape[1:])

        # init variables
        self.queries = 0
        hist = []
        find_flag = False

        # Initialization of the attack
        max_query_count_to_find_direction = self.max_query_count
        test_batch = 100
        assert max_query_count_to_find_direction % test_batch == 0
        while (not find_flag) and self.queries < max_query_count_to_find_direction:
            try_x_adv = target_x[torch.randint(0, target_x.shape[0], (test_batch,))]
            new_x_adv = torch.stack([self.image_augment(x_adv) for x_adv in try_x_adv])
            direction_sgn = torch.sign(new_x_adv - x)
            # test norm2
            d = torch.norm(torch.reshape(new_x_adv - x, (test_batch, -1)), p=2, dim=1)
            d = torch.reshape(d, (test_batch, 1, 1, 1))
            try_flag = self.is_adversarial(self.get_xadv(x, direction_sgn / torch.norm(direction_sgn[0], p=2).item(), d), y, target_y)
            if torch.sum(try_flag) > 0:
                # Find a good direction
                find_flag = True
                for i in range(test_batch):
                    hist.append(direction_sgn[i])
                    if try_flag[i]:
                        self.queries -= test_batch - i - 1
                        break
            else:
                hist += [x for x in direction_sgn]

        return torch.stack(hist, dim=0).numpy(), self.queries, find_flag

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
        assert self.targeted, "This method only supports targeted attack!"
        assert (target_x is not None) and (target_y is not None), "target_x and target_y inputs are required for testing initialization of GuidedRay Attack!"
        shape = list(x.shape)
        assert shape[0] == 1, "RayS Attack only supports batch size of 1 right now!"

        # init variables
        directions = []

        for i in range(num_directions):
            try_x_adv = target_x[torch.randint(0, target_x.shape[0], (1,))]
            new_x_adv = self.image_augment(try_x_adv[0])
            new_x_adv = torch.unsqueeze(new_x_adv, dim=0)
            direction_sgn = torch.sign(new_x_adv - x)
            directions.append(direction_sgn)

        return torch.concat(directions, dim=0).numpy()