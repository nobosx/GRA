import numpy as np
import torch
import math
from attackers.basic_attacker import BasicAttacker

class BounceAttacker(BasicAttacker):
    def __init__(self, model, norm_order = np.inf, targeted = False, max_query_count = 10000, early_stop = False, epsilon_metric = 0.01, verbose = True, stepsize_search: str = 'geometric_progression'):
        super().__init__(model, norm_order, targeted, max_query_count, early_stop, epsilon_metric, verbose)
        assert stepsize_search in ['geometric_progression', 'grid_search']
        self.stepsize_search = stepsize_search
        # Determin gamma value
        if model.dataset_name == 'imagenet':
            if self.norm_order == 2:
                self.gamma = 1000.0
            else:
                self.gamma = 10000.0
        else:
            if self.norm_order == 2:
                self.gamma = 1.0
            else:
                self.gamma = 100.0
        self.init_num_evals = 100
        self.lb = 0.0
        self.ub = 1.0
        self.max_num_evals = 10000
        self.max_bath_restraint = 50 if model.dataset_name == 'imagenet' else None

    def decision_function(self, image, y, target=None):
        """
        Decision function output True on the desired side of the boundary,
        False otherwise.
        """
        if len(image.shape) == 3:
            # Need to add a bacth_size dim
            image = torch.unsqueeze(image, 0)
        image = torch.clamp(image, self.lb, self.ub)
        if (self.max_bath_restraint is None) or (len(image) <= self.max_bath_restraint):
            return self.is_adversarial(image, y, target)
        else:
            # Only use limited GPU memory
            output_array = []
            batch_num = (len(image) + self.max_bath_restraint - 1) // self.max_bath_restraint
            for batch_index in range(batch_num):
                l = batch_index * self.max_bath_restraint
                r = min((batch_index + 1) * self.max_bath_restraint, len(image))
                output_array.append(self.is_adversarial(image[l:r], y, target))
            return torch.cat(output_array, dim=0)
    
    def initialize(self, sample, y, target=None, target_x=None):
        """ 
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0

        if not self.targeted:
            # Find a misclassified random noise.
            while True:
                random_noise = torch.rand_like(sample)
                success = self.decision_function(random_noise, y, target).item()
                num_evals += 1
                if success:
                    break
                if num_evals >= self.max_query_count:
                    return None
			
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success = self.decision_function(blended, y, target).item()
                if success:
                    high = mid
                else:
                    low = mid
            
            initialization = (1 - high) * sample + high * random_noise
        
        else:
            initialization = target_x
        
        return initialization
    
    def project(self, original_image, perturbed_images, alphas):
        alphas_shape = [len(alphas)] + [1] * len(original_image.shape)
        alphas = alphas.reshape(alphas_shape)
        if self.norm_order == 2:
            return (1-alphas) * original_image + alphas * perturbed_images
        else:
            out_images = torch.clamp(perturbed_images, original_image - alphas, original_image + alphas)
            return out_images
    
    def compute_distance(self, x_ori, x_pert):
        # Compute the distance between two images.
        return torch.norm(x_ori - x_pert, p=self.norm_order).item()
    
    def binary_search_batch(self, original_image, perturbed_images, y, target=None):
        """ Binary search to approach the boundar. """

        # Compute distance between each of perturbed image and original image.
        dists_post_update = torch.tensor([self.compute_distance(original_image, perturbed_image) for perturbed_image in perturbed_images], dtype=torch.float64)

        # Choose upper thresholds in binary searchs based on constraint.
        if self.norm_order == np.inf:
            highs = dists_post_update
            # Stopping criteria.
            thresholds = torch.minimum(dists_post_update * self.theta, torch.tensor(self.theta, dtype=torch.float64))
        else:
            highs = torch.ones(len(perturbed_images), dtype=torch.float64)
            thresholds = self.theta
        lows = torch.zeros(len(perturbed_images), dtype=torch.float64)

        # Call recursive function.
        while torch.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image, perturbed_images, mids.float())

            # Update highs and lows based on model decisions.
            decisions = self.decision_function(torch.reshape(mid_images, (mid_images.shape[0], mid_images.shape[2], mid_images.shape[3], mid_images.shape[4])), y, target)
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)
        
        out_images = self.project(original_image, perturbed_images, highs.float())

        # Compute distance of the output image to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = torch.tensor([self.compute_distance(original_image, out_image) for out_image in out_images])
        idx = torch.argmin(dists)

        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist
    
    def select_delta(self, dist_post_update):
        """ 
        Choose the delta at the scale of distance 
        between x and perturbed sample. 

        """
        if self.cur_iter == 1:
            delta = 0.1 * (self.ub - self.lb)
        else:
            if self.norm_order == 2:
                delta = math.sqrt(self.d) * self.theta * dist_post_update
            else:
                delta = self.d * self.theta * dist_post_update
        return delta
    
    def approximate_gradient(self, sample, num_evals, delta, y, target=None):
        # Generate random vectors
        noise_shape = [num_evals] + list(sample.shape)
        if self.norm_order == 2:
            rv = torch.randn(size=noise_shape)
        elif self.norm_order == np.inf:
            rv = torch.rand(size=noise_shape) * 2 - 1
        
        rv = torch.reshape(rv, (num_evals, -1))
        rv = rv / torch.norm(rv, p=2, dim=1, keepdim=True)
        rv = torch.reshape(rv, noise_shape)
        perturbed = sample + delta * rv
        perturbed = torch.clamp(perturbed, self.lb, self.ub)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions = self.decision_function(torch.reshape(perturbed, (num_evals, noise_shape[-3], noise_shape[-2], noise_shape[-1])), y, target)
        decision_shape = [len(decisions)] + [1] * len(sample.shape)
        fval = 2 * decisions.float().reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if torch.mean(fval) == 1.0:
            gradf = torch.mean(rv, axis=0)
        elif torch.mean(fval) == -1.0:
            gradf = - torch.mean(rv, axis=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, axis=0)
        
        # Get the gradient direction.
        gradf = gradf / torch.norm(gradf)

        return gradf
    
    def geometric_progression_for_stepsize(self, x, update, dist, y, target=None):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching 
        the desired side of the boundary,
        """
        epsilon = dist / math.sqrt(self.cur_iter)

        def phi(epsilon):
            new = x + epsilon * update
            success = self.decision_function(new, y, target).item()
            return success

        while not phi(epsilon):
            epsilon /= 2.0
        
        return epsilon
    
    def gradient_orthogonal_component(self, gradf, ori_image, bound_image):
        # Compute x to x'
        d = bound_image - ori_image
        d_unit = d / (torch.norm(d).item() + 1e-10)

        # Compute the projection of gradf on d_unit
        projection_length = torch.sum(gradf * d_unit)
        projection_vector = projection_length * d_unit

        # Compute orth component
        orth_component = gradf - projection_vector

        return orth_component

    def bounce_direction(self, gradf, ori_image, bound_image, last_direction):
        miu = 0.5
        orth = self.gradient_orthogonal_component(gradf, ori_image, bound_image)
        bounce = orth + miu * last_direction
        bounce_unit = bounce / (torch.norm(bounce).item() + 1e-10)
        return bounce_unit
    
    def smooth(self, x, xl, xp, eps=1e-8):
        d1 = x - xp
        d2 = xl - xp
        smooth = (torch.norm(d1).item() + eps) / (torch.norm(d2).item() + eps)
        return smooth

    def attack(self, x, y, target_x=None, target_y=None):
        assert (not self.targeted) or (target_y is not None), "An target_y (target label) must be provided in a targeted attack!"
        assert (not self.targeted) or (target_x is not None), "An target_x (auxiliary target image) must be provided in a targeted attack!"
        if self.targeted:
            # Only use the first target_x
            first_target_x = torch.unsqueeze(target_x[0], dim=0)
        else:
            first_target_x = None

        # Set binary search threshold.
        self.queries = 0
        self.d = int(np.prod(x.shape))
        if self.norm_order == 2:
            self.theta = self.gamma / (math.sqrt(self.d) * self.d)
        else:
            self.theta = self.gamma / (self.d ** 2)
        
        # Initialize
        perturbed = self.initialize(x, y, target_y, first_target_x)

        if perturbed is None:
            hist = [(self.max_query_count, np.inf)]
            return None, np.inf, hist

        # Bounce params
        c = 0
        last_direction = torch.zeros_like(x)

        # Project the initialization to the boundary
        perturbed, dist_post_update = self.binary_search_batch(x, torch.unsqueeze(perturbed, 0), y, target_y)
        dist = self.compute_distance(perturbed, x)
        hist = [(self.queries, dist)]
        assert dist > 0 and dist < 1

        for j in range(self.max_query_count):
            self.cur_iter = j + 1
            last_perturbed = perturbed

            # Choose delta.
            delta = self.select_delta(dist_post_update)

            # Choose number of evaluations.
            num_evals = int(self.init_num_evals * math.sqrt(j + 1))
            num_evals = int(min(num_evals, self.max_num_evals))

            # approximate gradient.
            gradf = self.approximate_gradient(perturbed, num_evals, delta, y, target_y)

            # when c < 3, use the bounce direction
            if c < 3:
                direct = self.bounce_direction(gradf, x, perturbed, last_direction)
            else:
                direct = gradf

            last_direction = direct

            if self.norm_order == np.inf:
                update = torch.sign(direct)
            else:
                update = direct
            
            # search step size
            # print("Begin search step size")
            if self.stepsize_search == 'geometric_progression':
                # find step size
                epsilon = self.geometric_progression_for_stepsize(perturbed, update, dist, y, target_y)

                # update the sample
                perturbed = torch.clamp(perturbed + epsilon * update, self.lb, self.ub)
                # Exit this attack
                if not self.decision_function(perturbed, y, target_y):
                    perturbed = last_perturbed
                    self.queries -= 1
                    break

                # Binary search to return to the boundary.
                perturbed, dist_post_update = self.binary_search_batch(x, torch.unsqueeze(perturbed, 0), y, target_y)
                if not self.decision_function(perturbed, y, target_y):
                    perturbed = last_perturbed
                    self.queries -= 1
                    break
                
            else:
                # Grid search for stepsize.
                epsilons = torch.tensor(np.logspace(-4, 0, num=20, endpoint = True) * dist)
                epsilons_shape = [20] + len(x.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = torch.clamp(perturbeds, self.lb, self.ub)
                idx_perturbed = self.decision_function(torch.reshape(perturbeds, (20, perturbeds.shape[-3], perturbeds.shape[-2], perturbeds.shape[-1])), y, target_y)

                if torch.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update = self.binary_search_batch(x, perturbeds[idx_perturbed], y, target_y)
            if self.smooth(perturbed, last_perturbed, x) > 0.99:
                c += 1
            else:
                c = 0

            # compute new distance
            dist = self.compute_distance(perturbed, x)

            # There is a strange bug!
            # Sometimes dist == nan
            if not (dist > 0 and dist < 1):
                break

            if dist > hist[-1][1]:
                dist = hist[-1][1]

            hist.append((self.queries, dist))

            if self.queries >= self.max_query_count:
                break
            # Consider early stop technique
            if self.early_stop and (dist < self.epsilon_metric):
                if self.verbose:
                    print(f"Early stop at {self.queries} queries!")
                break
        
        return perturbed, dist, hist
