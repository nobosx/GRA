from attackers.basic_attacker import BasicAttacker
import torch
import math
import numpy as np

class EllipsoidTangentFinder(object):
    def __init__(self, x, ball_center, short_radius, long_radius, plane_normal_vector, norm="l2"):
        '''
        :param x: the original image which is outside the ball, vector form.
        :param ball_center: the perturbed image lies on the decision boundary
        :param short_radius: the radius of the ball
        :param plane_normal_vector: the normal vector of hyperplane
        :param norm: l2 or linf
        '''
        self.x = x
        self.o = ball_center
        self.ox = self.x - self.o
        self.S = short_radius
        self.L = long_radius

        self.norm = norm
        self.u = plane_normal_vector # unit normal vector
        self.ord = np.inf if self.norm == "linf" else 2

    def compute_2D_x(self):
        x_norm = torch.norm(self.ox, p=self.ord)
        theta = self.theta()
        return (x_norm * torch.sin(theta), -x_norm * torch.cos(theta))

    def compute_tangent_point_of_ellipse(self):
        S = self.S
        L = self.L
        x_2D = self.compute_2D_x()
        x0 = x_2D[0].item()
        z0 = x_2D[1].item()
        in_sqrt = -L**2 * S**2 + L**2 * x0**2 + S**2 * z0**2
        if in_sqrt < 0:
            in_sqrt = 0

        # Below is NIPS version!
        xk = S**2 * (L**2 - z0 * (L**2 * S**2 * z0/ (L**2 * x0**2 + S**2 * z0**2) + L**2 * x0 * math.sqrt(in_sqrt)/(L**2 * x0**2 + S**2 * z0**2))) / (L**2 * x0)
        zk = (L**2 * S**2 * z0)/(L**2 * x0 ** 2 + S**2 * z0**2) + (L**2 * x0 * math.sqrt(in_sqrt))/(L**2 * x0 ** 2 + S**2 * z0**2)

        return xk, zk

    def theta(self):
        return torch.acos(torch.dot(self.ox, -self.u)/(torch.norm(self.ox, p=self.ord) * torch.norm(self.u, p=self.ord)))

    def compute_tangent_point(self):
        x_k, z_k = self.compute_tangent_point_of_ellipse()
        numerator = self.ox - torch.dot(self.ox, self.u) * self.u / torch.norm(self.u) ** 2
        ok_prime = (numerator / torch.norm(numerator, p=self.ord)) * math.fabs(x_k)
        ok = ok_prime + z_k * self.u # / torch.norm(self.u)
        return ok + self.o

class TangentAttacker(BasicAttacker):
    def __init__(self, model, norm_order = np.inf, targeted = False, max_query_count = 10000, early_stop = False, epsilon_metric = 0.01, verbose = True):
        super().__init__(model, norm_order, targeted, max_query_count, early_stop, epsilon_metric, verbose)
        self.clip_min = 0.0
        self.clip_max = 1.0
        self.init_num_evals = 100
        self.max_num_evals = 10000
        self.stepsize_search = 'geometric_progression'
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
        self.max_bath_restraint = 50 if model.dataset_name == 'imagenet' else None
        self.radius_ratio = 1.1 if model.dataset_name == 'imagenet' else 1.5


    def decision_function(self, images, true_labels, target_labels):
        assert len(images.shape) == 4
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max)
        if (self.max_bath_restraint is None) or (len(images) <= self.max_bath_restraint):
            return self.is_adversarial(images, true_labels, target_labels)
        else:
            # Only use limited GPU memory
            output_array = []
            batch_num = (len(images) + self.max_bath_restraint - 1) // self.max_bath_restraint
            for batch_index in range(batch_num):
                l = batch_index * self.max_bath_restraint
                r = min((batch_index + 1) * self.max_bath_restraint, len(images))
                output_array.append(self.is_adversarial(images[l:r], true_labels, target_labels))
            return torch.cat(output_array, dim=0)

    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=sample.shape)).float()
                success = self.decision_function(random_noise[None], true_labels, target_labels)[0].item()
                if success:
                    break
                if self.queries >= self.max_query_count:
                    print("Initialization failed! Success failed!")
                    return None
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success = self.decision_function(blended[None], true_labels, target_labels)[0].item()
                if success:
                    high = mid
                else:
                    low = mid
            initialization = (1 - high) * sample + high * random_noise
        else:
            initialization = target_images
        return initialization
    
    def compute_distance(self, x_ori, x_pert, norm=2):
        # Compute the distance between two images.
        if norm == 2:
            return torch.norm(x_ori - x_pert,p=2).item()
        elif norm == np.inf:
            return torch.max(torch.abs(x_ori - x_pert)).item()
        
    def clip_image(self, image, clip_min, clip_max):
        # Clip an image, or an image batch, with upper and lower threshold.
        return torch.min(torch.max(image, clip_min), clip_max)

    def project(self, original_image, perturbed_images, alphas):
        alphas_shape = [alphas.size(0)] + [1] * len(original_image.shape)
        alphas = alphas.view(*alphas_shape)
        if self.norm_order == 2:
            return (1 - alphas) * original_image + alphas * perturbed_images
        elif self.norm_order == np.inf:
            out_images = self.clip_image(perturbed_images, original_image - alphas, original_image + alphas)
            return out_images
    
    def binary_search_batch(self, original_image, perturbed_images, true_labels, target_labels):
        assert len(perturbed_images.shape) == 4 and perturbed_images.shape[1:] == original_image.shape
        # Compute distance between each of perturbed image and original image.
        dists_post_update = torch.tensor([
            self.compute_distance(
                original_image,
                perturbed_image,
                self.norm_order
            ) for perturbed_image in perturbed_images], dtype=torch.float64)
        # Choose upper thresholds in binary searchs based on constraint.
        if self.norm_order == np.inf:
            highs = dists_post_update
            thresholds = torch.clamp_max(dists_post_update * self.theta, max=self.theta)
            assert thresholds.dtype == torch.float64
        else:
            highs = torch.ones(perturbed_images.size(0), dtype=torch.float64)
            thresholds = self.theta
        lows = torch.zeros(perturbed_images.size(0), dtype=torch.float64)

        while torch.max((highs - lows) / thresholds).item() > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_images = self.project(original_image, perturbed_images, mids.float())
            # Update highs and lows based on model decisions.
            decisions = self.decision_function(mid_images, true_labels, target_labels)
            decisions = decisions.int()
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)
        out_images = self.project(original_image, perturbed_images, highs.float())
        dists = torch.tensor([
            self.compute_distance(
                original_image,
                out_image,
                self.norm_order
            ) for out_image in out_images])
        idx = torch.argmin(dists)
        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist
    
    def select_delta(self, cur_iter, dist_post_update):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.

        """
        if cur_iter == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.norm_order == 2:
                delta = np.sqrt(self.dim) * self.theta * dist_post_update
            else:
                delta = self.dim * self.theta * dist_post_update
        return delta
    
    def approximate_gradient(self, sample, true_labels, target_labels, num_evals, delta):
        clip_max, clip_min = self.clip_max, self.clip_min

        # Generate random vectors.
        noise_shape = [num_evals] + list(sample.shape)
        if self.norm_order == 2:
            rv = torch.randn(*noise_shape)
        elif self.norm_order == np.inf:
            rv = torch.from_numpy(np.random.uniform(low=-1, high=1, size=noise_shape)).float()
        rv = rv / torch.sqrt(torch.sum(torch.mul(rv,rv), dim=(1,2,3),keepdim=True))
        perturbed = sample + delta * rv
        perturbed = torch.clamp(perturbed, clip_min, clip_max)
        rv = (perturbed - sample) / delta

        decisions = self.decision_function(perturbed, true_labels, target_labels)
        decision_shape = [decisions.size(0)] + [1] * len(sample.shape)
        fval = 2 * decisions.float().view(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if torch.mean(fval).item() == 1.0:
            gradf = torch.mean(rv, dim=0)
        elif torch.mean(fval).item() == -1.0:
            gradf = -torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, dim=0)
        
        # Get the gradient direction.
        gradf = gradf / torch.norm(gradf,p=2)

        return gradf
    
    def geometric_progression_for_tangent_point(self, x_original, x_boundary, normal_vector, true_labels, target_labels, dist, cur_iter):
        """
        Geometric progression to search for stepsize.
        Keep decreasing stepsize by half until reaching
        the desired side of the boundary,
        """
        long_radius = dist / np.sqrt(cur_iter)
        short_radius = long_radius / self.radius_ratio
        while True:
            tangent_finder = EllipsoidTangentFinder(x_original.view(-1), x_boundary.view(-1),  short_radius, long_radius, normal_vector.view(-1), norm="l2")
            tangent_point = tangent_finder.compute_tangent_point()
            tangent_point = tangent_point.view_as(x_original).type(x_original.dtype)

            success = self.decision_function(tangent_point[None], true_labels, target_labels)[0].item()
            if bool(success):
                break
            long_radius /= 2.0
            short_radius = long_radius / self.radius_ratio
        tangent_point = torch.clamp(tangent_point, self.clip_min, self.clip_max)
        return tangent_point

    def attack(self, x, y, target_x=None, target_y=None):
        assert (not self.targeted) or (target_y is not None), "An target_y (target label) must be provided in a targeted attack!"
        assert (not self.targeted) or (target_x is not None), "An target_x (auxiliary target image) must be provided in a targeted attack!"
        assert x.size(0) == 1, "Tangent attack only supports one sample at one time!"
        if self.targeted:
            # Only use the first target_x
            target_images = target_x[0]
        else:
            target_images = None
        # Rename some vars
        true_labels = y
        target_labels = target_y
        batch_size = x.size(0)
        images = torch.squeeze(x, dim=0)
        # Init some attack params
        self.dim = np.prod(list(images.shape))
        if self.norm_order == 2:
            self.theta = self.gamma / (np.sqrt(self.dim) * self.dim)
        elif self.norm_order == np.inf:
            self.theta = self.gamma / (self.dim ** 2)
        # Init query and hist
        self.queries = 0
        hist = [(0, np.inf)]

        # Initialize.
        perturbed = self.initialize(images, target_images, true_labels, target_labels)
        if perturbed is None:
            hist.append((self.queries, np.inf))
            return None, np.inf, hist
        dist = torch.norm((perturbed - images).view(batch_size, -1), self.norm_order, 1)
        hist.append((self.queries, dist.item()))

        # Project the initialization to the boundary.
        perturbed, dist_post_update = self.binary_search_batch(images, perturbed[None], true_labels,target_labels)
        assert self.is_adversarial(perturbed, y, target_y).item()
        self.queries -= 1
        dist =  torch.norm((perturbed - images).view(batch_size, -1), self.norm_order, 1).item()
        hist.append((self.queries, dist))

        cur_iter = 0
        for j in range(self.max_query_count):
            last_perturbed = perturbed
            cur_iter += 1
            # Choose delta.
            delta = self.select_delta(cur_iter, dist_post_update)
            # Choose number of evaluations.
            num_evals = int(self.init_num_evals * np.sqrt(j+1))
            num_evals = int(min([num_evals, self.max_num_evals]))
            # approximate gradient
            gradf = self.approximate_gradient(perturbed, true_labels, target_labels, num_evals, delta)
            if self.norm_order == np.inf:
                gradf = torch.sign(gradf)
            
            # search step size.
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                perturbed_Tagent = self.geometric_progression_for_tangent_point(images, perturbed, gradf, true_labels, target_labels, dist, cur_iter)
                perturbed = perturbed_Tagent
                # Exit this attack
                if not self.is_adversarial(perturbed, y, target_y).item():
                    perturbed = last_perturbed
                    self.queries -= 1
                    break

                perturbed, dist_post_update = self.binary_search_batch(images, perturbed[None], true_labels, target_labels)
                # Exit this attack
                if not self.is_adversarial(perturbed, y, target_y).item():
                    perturbed = last_perturbed
                    self.queries -= 1
                    break

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                update = gradf
                epsilons = torch.logspace(-4, 0, steps=20) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.view(epsilons_shape) * update
                perturbeds = torch.clamp(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = self.decision_function(perturbeds, true_labels, target_labels)
                if idx_perturbed.int().sum().item() > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update = self.binary_search_batch(images, perturbeds[idx_perturbed], true_labels, target_labels)
            
            dist =  torch.norm((perturbed - images).view(batch_size, -1), self.norm_order, 1).item()
            if dist > hist[-1][1]:
                dist = hist[-1][1]
            hist.append((self.queries, dist))

            if self.queries >= self.max_query_count:
                break

            if self.early_stop and dist < self.epsilon_metric:
                print(f"Early stop at {self.queries} queries!")
                break
            
            if dist < 1e-4:
                break
        
        return perturbed, dist, hist