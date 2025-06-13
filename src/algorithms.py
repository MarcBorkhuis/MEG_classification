"""Domain Adaptation and Regularization Algorithms

This module implements various domain adaptation and regularization algorithms for MEG classification:
- ERM: Empirical Risk Minimization (standard training)
- IRM: Invariant Risk Minimization
- Mixup: Data augmentation through interpolation
- GroupDRO: Group Distributionally Robust Optimization
- CORAL: Correlation Alignment
- MMD: Maximum Mean Discrepancy

Each algorithm is designed to handle domain shifts in MEG data, particularly
for cross-subject classification scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from itertools import combinations
from utils import random_pairs_of_minibatches

# List of available algorithms
ALGORITHMS = ["ERM", "IRM", "Mixup", "GroupDRO", "CORAL", "MMD"]

def get_algorithm_class(algorithm_name: str) -> type:
    """Get algorithm class by name."""
    if algorithm_name not in globals():
        raise NotImplementedError(f"Algorithm {algorithm_name} not implemented")
    return globals()[algorithm_name]

class Algorithm(nn.Module):
    """Base class for all training algorithms."""
    def __init__(self, network: nn.Module, hparams: dict, device: torch.device):
        super().__init__()
        self.network = network
        self.hparams = hparams
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams.get('lr', 1e-3),
            weight_decay=hparams.get('weight_decay', 0)
        )

    def update(self, minibatches: list, unlabeled=None) -> dict:
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ERM(Algorithm):
    """Empirical Risk Minimization (standard training)."""
    def update(self, minibatches: list, unlabeled=None) -> dict:
        all_x = torch.cat([x for x, y in minibatches]).to(self.device)
        all_y = torch.cat([y for x, y in minibatches]).long().to(self.device)
        
        loss = F.cross_entropy(self.predict(all_x), all_y)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item()}

class IRM(ERM):
    """Invariant Risk Minimization."""
    def __init__(self, network, hparams, device):
        super().__init__(network, hparams, device)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, y)
        g = grad(loss, [scale], create_graph=True)[0]
        return torch.sum(g**2)

    def update(self, minibatches: list, unlabeled=None) -> dict:
        penalty_weight = self.hparams.get("irm_lambda", 1.0) if self.update_count >= self.hparams.get("irm_penalty_anneal_iters", 0) else 1.0
        
        nll = 0.0
        penalty = 0.0
        for x, y in minibatches:
            x, y = x.to(self.device), y.long().to(self.device)
            logits = self.predict(x)
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + penalty_weight * penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}

class Mixup(ERM):
    """Mixup data augmentation."""
    def update(self, minibatches: list, unlabeled=None) -> dict:
        all_x = torch.cat([d[0] for d in minibatches]).to(self.device)
        all_y = torch.cat([d[1] for d in minibatches]).long().to(self.device)
        
        alpha = self.hparams.get('mixup_alpha', 0.2)
        lam = np.random.beta(alpha, alpha)
        
        batch_size = all_x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * all_x + (1 - lam) * all_x[index, :]
        y_a, y_b = all_y, all_y[index]
        
        logits = self.predict(mixed_x)
        loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item()}

class GroupDRO(Algorithm):
    """Group Distributionally Robust Optimization."""
    def __init__(self, network: nn.Module, hparams: dict, device: torch.device):
        super().__init__(network, hparams, device)
        self.group_weights = None

    def initialize_group_weights(self, n_groups: int):
        self.group_weights = torch.ones(n_groups, device=self.device) / n_groups

    def update(self, minibatches: list, unlabeled=None) -> dict:
        if self.group_weights is None:
             raise RuntimeError("Group weights not initialized. Call initialize_group_weights first.")
        
        group_losses = []
        for x, y in minibatches:
            x, y = x.to(self.device), y.long().to(self.device)
            logits = self.predict(x)
            group_losses.append(F.cross_entropy(logits, y))
        
        group_losses = torch.stack(group_losses)
        
        # Update and use weights
        self.group_weights *= torch.exp(self.hparams['dro_eta'] * group_losses.detach())
        self.group_weights /= self.group_weights.sum()
        
        loss = (group_losses * self.group_weights).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # FIX: Do not return the group_weights tensor which requires grad
        return {'loss': loss.item()}

class CORAL(Algorithm):
    """Correlation Alignment."""
    def coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = source.size(1)
        source_centered = source - source.mean(0, keepdim=True)
        target_centered = target - target.mean(0, keepdim=True)
        source_cov = (source_centered.t() @ source_centered) / (source_centered.size(0) - 1)
        target_cov = (target_centered.t() @ target_centered) / (target_centered.size(0) - 1)
        return torch.sum((source_cov - target_cov)**2) / (4 * d**2)

    def update(self, minibatches: list, unlabeled=None) -> dict:
        pred_loss = 0
        feats = []
        for x, y in minibatches:
            x, y = x.to(self.device), y.long().to(self.device)
            feat = self.network.get_features(x)
            feats.append(feat)
            logits = self.network.classif(feat)
            pred_loss += F.cross_entropy(logits, y)
        pred_loss /= len(minibatches)
        
        coral_loss = torch.tensor(0.0, device=self.device)
        if len(minibatches) > 1:
            for f1, f2 in combinations(feats, 2):
                coral_loss += self.coral_loss(f1, f2)
        
        loss = pred_loss + self.hparams.get('coral_lambda', 1.0) * coral_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item(), 'pred_loss': pred_loss.item(), 'coral': coral_loss.item()}

class MMD(Algorithm):
    """Maximum Mean Discrepancy."""
    def _rbf_kernel(self, X, Y, sigma_list):
        dist = torch.cdist(X, Y, p=2)**2
        s = torch.tensor(sigma_list, device=X.device).view(-1, 1, 1)
        return torch.exp(-dist / (2 * s**2)).sum(0)
    
    def mmd_loss(self, source, target):
        sigma_list = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        k_ss = self._rbf_kernel(source, source, sigma_list).mean()
        k_tt = self._rbf_kernel(target, target, sigma_list).mean()
        k_st = self._rbf_kernel(source, target, sigma_list).mean()
        return k_ss + k_tt - 2 * k_st

    def update(self, minibatches: list, unlabeled=None) -> dict:
        pred_loss = 0
        feats = []
        for x, y in minibatches:
            x, y = x.to(self.device), y.long().to(self.device)
            feat = self.network.get_features(x)
            feats.append(feat)
            logits = self.network.classif(feat)
            pred_loss += F.cross_entropy(logits, y)
        pred_loss /= len(minibatches)
        
        mmd_term = torch.tensor(0.0, device=self.device)
        if len(minibatches) > 1:
            for f1, f2 in combinations(feats, 2):
                mmd_term += self.mmd_loss(f1, f2)

        loss = pred_loss + self.hparams.get('mmd_lambda', 1.0) * mmd_term
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item(), 'pred_loss': pred_loss.item(), 'mmd': mmd_term.item()}
