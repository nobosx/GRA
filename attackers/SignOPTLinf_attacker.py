from attackers.basic_attacker import BasicAttacker
import torch
import numpy as np

class SignOPTLinfAttacker(BasicAttacker):
    def __init__(self, model, norm_order = np.inf, targeted = False, max_query_count = 10000, early_stop = False, epsilon_metric = 0.01, verbose = True):
        super().__init__(model, norm_order, targeted, max_query_count, early_stop, epsilon_metric, verbose)
        assert norm_order == np.inf, "SignOPTLinf attacker only supports Linf attack!"
        self.alpha = 0.2
        self.beta = 0.001
        self.k = 200
        self.tol = 1e-4 if self.model.dataset_name == 'imagenet' else (self.beta / 500)

    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        if initial_lbd > current_best:
            if not self.is_adversarial(x0 + current_best * theta, y0).item():
                return float('inf')
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0
        count = 0
        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            count += 1
            if self.is_adversarial(x0 + lbd_mid * theta, y0).item():
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if count >= 200:
                print("Break in the first fine_grained_binary_search!")
                break
        return lbd_hi
    
    def fine_grained_binary_search_targeted(self, x0, t, theta, initial_lbd, current_best):
        if initial_lbd > current_best:
            if not self.is_adversarial(x0 + current_best * theta, None, t).item():
                return float('inf')
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0
        count = 0
        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            if not self.is_adversarial(x0 + lbd_mid * theta, None, t).item():
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
            if count >= 200:
                print("Break in the first fine_grained_binary_search!")
                break
        return lbd_hi
    
    def sign_grad_v1(self, images, true_label, theta, initial_lbd, h=0.001, target_label=None):
        K = self.k
        images_batch = []
        u_batch = []
        for iii in range(K):
            u = torch.randn_like(theta)
            u /= torch.norm(u.flatten(), p=float('inf'))
            new_theta = theta + h * u
            new_theta /= torch.norm(new_theta.flatten(), p=float('inf'))
            u_batch.append(u)
            images_batch.append(images + initial_lbd * new_theta)
        images_batch = torch.cat(images_batch, 0)
        u_batch = torch.cat(u_batch, 0)  # B,C,H,W
        assert u_batch.dim() == 4
        sign = torch.ones(K)
        if target_label is not None:
            target_labels = torch.tensor([target_label for _ in range(K)]).long()
            predict_labels = self.query_model_hard_label(images_batch)
            sign[predict_labels == target_labels] = -1
        else:
            true_labels = torch.tensor([true_label for _ in range(K)]).long()
            predict_labels = self.query_model_hard_label(images_batch)
            sign[predict_labels != true_labels] = -1
        sign_grad = torch.sum(u_batch * sign.view(K, 1, 1, 1), dim=0, keepdim=True)
        sign_grad = sign_grad / K

        return sign_grad
    
    def fine_grained_binary_search_local(self, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
        lbd = initial_lbd
        tol = max(tol, 2e-9)
        # still inside boundary
        if not self.is_adversarial(x0 + lbd * theta, y0).item():
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            while not self.is_adversarial(x0 + lbd_hi * theta, y0).item():
                lbd_hi = lbd_hi * 1.01
                if lbd_hi > 20:
                    return float('inf')
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            while self.is_adversarial(x0 + lbd_lo * theta, y0).item():
                lbd_lo = lbd_lo * 0.99
        tot_count = 0
        while (lbd_hi - lbd_lo) > tol:
            tot_count += 1
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            if self.is_adversarial(x0 + lbd_mid * theta, y0).item():
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if tot_count > 200:
                print("Reach max while limit, maybe dead loop in binary search function, break!")
                break
        return lbd_hi
    
    def fine_grained_binary_search_local_targeted(self, x0, t, theta, initial_lbd=1.0, tol=1e-5):
        lbd = initial_lbd
        tol = max(tol, 2e-9)
        if not self.is_adversarial(x0 + lbd * theta, None, t).item():
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            while not self.is_adversarial(x0 + lbd_hi * theta, None, t).item():
                lbd_hi = lbd_hi * 1.01
                if lbd_hi > 100:
                    return float('inf')
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            while self.is_adversarial(x0 + lbd_lo * theta, None, t).item():
                lbd_lo = lbd_lo * 0.99
        tot_count = 0
        while (lbd_hi - lbd_lo) > tol:
            tot_count += 1
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            if self.is_adversarial(x0 + lbd_mid * theta, None, t).item():
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if tot_count > 200:
                print("reach max while limit, dead loop in binary search function, break!")
                break
        return lbd_hi

    def targeted_attack(self, images, true_labels, target_labels, target_class_image):
        # Init the attack
        self.queries = 0
        target_label = target_labels[0].item()
        alpha = self.alpha
        beta = self.beta
        best_theta, g_theta = None, float('inf')
        hist = [(0, np.inf)]

        xi = target_class_image
        theta = xi - images
        initial_lbd = torch.norm(theta, float('inf'))
        theta /= initial_lbd
        lbd = self.fine_grained_binary_search_targeted(images, target_label, theta, initial_lbd, g_theta)
        best_theta, g_theta = theta, lbd
        if g_theta == np.inf:
            print("Attack couldn't find valid initial, failed!")
            return images, np.inf, hist
        print("Find best dist {:.4f} using {} queries.".format(g_theta, self.queries))
        dist = torch.norm(best_theta * g_theta, p=float('inf')).item()
        hist.append((self.queries, dist))

        # Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        for i in range(self.max_query_count):
            sign_gradient = self.sign_grad_v1(images, None, xg, initial_lbd=gg, h=beta, target_label=target_label)
            # Line search
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= torch.norm(new_theta.flatten(), float('inf'))
                tol = self.tol
                new_g2 = self.fine_grained_binary_search_local_targeted(images, target_label, new_theta, initial_lbd=min_g2, tol=tol)
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    dist = torch.norm(min_theta * min_g2, p=float('inf')).item()
                    hist.append((self.queries, dist))
                else:
                    break
            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= torch.norm(new_theta.flatten(), float('inf'))
                    tol = self.tol
                    new_g2 = self.fine_grained_binary_search_local_targeted(images, target_label, new_theta, initial_lbd=min_g2, tol=tol)
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        dist = torch.norm(min_theta * min_g2, p=float('inf')).item()
                        hist.append((self.queries, dist))
                        break
            if alpha < 1e-4:
                print("Warning: not moving!")
                beta = beta * 0.1
                if beta < 1e-8:
                    break
            xg, g2 = min_theta, min_g2
            gg = g2
            dist = torch.norm(gg * xg, p=float('inf')).item()
            hist.append((self.queries, dist))

            if self.queries >= self.max_query_count:
                break

            if self.early_stop and dist < self.epsilon_metric:
                print(f"Early stop at {self.queries} queries!")
                break

        assert self.is_adversarial(images + gg * xg, None, target_labels).item()
        
        assert dist - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, dist)
        return images + gg * xg, dist, hist
        

    def untargted_attack(self, images, true_labels):
        # Init the attack
        self.queries = 0
        alpha = self.alpha
        beta = self.beta
        ls_total = 0
        true_label = true_labels[0].item()
        hist = [(0, np.inf)]

        # Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        for i in range(num_directions):
            theta = torch.randn_like(images)
            if self.is_adversarial(images + theta, true_labels).item():
                initial_lbd = torch.norm(theta, p=float('inf'))
                theta /= initial_lbd
                lbd = self.fine_grained_binary_search(images, true_label, theta, initial_lbd, g_theta)
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
        if g_theta == float('inf'):
            print("Attacker couldn't find valid initial, failed!")
            return images, np.inf, hist
        print("Find best dist {:.4f} using {} queries.".format(g_theta, self.queries))
        dist = torch.norm(best_theta * g_theta, p=float('inf')).item()
        hist.append((self.queries, dist))

        # Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        for i in range(self.max_query_count):
            sign_gradient = self.sign_grad_v1(images, true_label, xg, initial_lbd=gg, h=beta)

            # Line search of the step size of gradient descent
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= torch.norm(new_theta.flatten(), p=float('inf'))
                tol = self.tol
                new_g2 = self.fine_grained_binary_search_local(images, true_label, new_theta, initial_lbd=min_g2, tol=tol)
                alpha = alpha * 2  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break
            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= torch.norm(new_theta.flatten(), float('inf'))
                    tol = self.tol
                    new_g2 = self.fine_grained_binary_search_local(images, true_label, new_theta, initial_lbd=min_g2, tol=tol)
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break
            if alpha < 1e-6:
                alpha = 1.0
                print("Warns: not moving!")
                beta = beta * 0.1
                if beta < 1e-8:
                    break
            xg, g2 = min_theta, min_g2
            gg = g2
            dist = torch.norm(gg * xg, p=float('inf')).item()
            hist.append((self.queries, dist))

            if self.queries >= self.max_query_count:
                break

            if self.early_stop and dist < self.epsilon_metric:
                print(f"Early stop at {self.queries} queries!")
                break

        assert self.is_adversarial(images + gg * xg, true_labels).item()

        assert dist - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, dist)
        return images + gg * xg, dist, hist

    def attack(self, x, y, target_x=None, target_y=None):
        assert (not self.targeted) or (target_y is not None), "An target_y (target label) must be provided in a targeted attack!"
        assert (not self.targeted) or (target_x is not None), "An target_x (auxiliary target image) must be provided in a targeted attack!"
        assert len(x) == 1, "Now Sign-OPT attacker only supports attacking one image at one time!"

        if self.targeted:
            # Only use the first target image
            first_target_x = torch.unsqueeze(target_x[0], dim=0)
            return self.targeted_attack(x, y, target_y, first_target_x)
        else:
            return self.untargted_attack(x, y)