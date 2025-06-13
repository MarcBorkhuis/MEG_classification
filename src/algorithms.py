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
    """Get algorithm class by name.
    
    Args:
        algorithm_name (str): Name of the algorithm
        
    Returns:
        type: The requested algorithm class
        
    Raises:
        NotImplementedError: If algorithm is not found
    """
    if algorithm_name not in globals():
        raise NotImplementedError(f"Algorithm {algorithm_name} not implemented")
    return globals()[algorithm_name]

class Algorithm:
    """Base class for all training algorithms.
    
    This class provides a common interface for all training algorithms,
    with methods for updating model parameters and making predictions.
    
    Args:
        network (nn.Module): Neural network model
        hparams (dict): Hyperparameters for the algorithm
        device (torch.device): Device to run the algorithm on
    """
    def __init__(self, network: nn.Module, hparams: dict, device: torch.device):
        self.network = network
        self.hparams = hparams
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay']
        )

    def update(self, minibatches: list) -> dict:
        """Update model parameters using the algorithm.
        
        Args:
            minibatches (list): List of (x, y) tuples for each domain
            
        Returns:
            dict: Dictionary containing loss values
        """
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions for input data.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Model predictions
        """
        return self.network(x)

class ERM(Algorithm):
    """Empirical Risk Minimization (standard training).
    
    This algorithm minimizes the average loss across all domains,
    treating them as a single distribution.
    """
    def update(self, minibatches: list) -> dict:
        """Update model parameters using ERM.
        
        Args:
            minibatches (list): List of (x, y) tuples for each domain
            
        Returns:
            dict: Dictionary containing loss values
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        all_x = all_x.to(self.device)
        all_y = all_y.to(self.device)
        
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(all_x), all_y)
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

class IRM(Algorithm):
    """Invariant Risk Minimization.
    
    This algorithm learns representations that are invariant across domains
    by penalizing the variance of the risk across domains.
    """
    def update(self, minibatches: list) -> dict:
        """Update model parameters using IRM.
        
        Args:
            minibatches (list): List of (x, y) tuples for each domain
            
        Returns:
            dict: Dictionary containing loss values
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        all_x = all_x.to(self.device)
        all_y = all_y.to(self.device)
        
        self.optimizer.zero_grad()
        
        features = self.network.get_features(all_x)
        logits = self.network.classif(features)
        
        loss = F.cross_entropy(logits, all_y)
        
        # IRM penalty
        scale = torch.tensor(1.).to(self.device).requires_grad_()
        loss_erm = F.cross_entropy(logits * scale, all_y)
        grad_1 = grad(loss_erm, [scale], create_graph=True)[0]
        loss_irm = torch.sum(grad_1 ** 2)
        
        loss = loss + self.hparams['irm_lambda'] * loss_irm
        
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item(), 'irm_penalty': loss_irm.item()}

class Mixup(Algorithm):
    """Mixup data augmentation.
    
    This algorithm performs data augmentation by interpolating between
    samples and their labels, encouraging smoother decision boundaries.
    """
    def update(self, minibatches: list) -> dict:
        """Update model parameters using Mixup.
        
        Args:
            minibatches (list): List of (x, y) tuples for each domain
            
        Returns:
            dict: Dictionary containing loss values
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        all_x = all_x.to(self.device)
        all_y = all_y.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Generate mixing weights
        alpha = self.hparams['mixup_alpha']
        lam = np.random.beta(alpha, alpha)
        
        # Shuffle data
        index = torch.randperm(all_x.size(0))
        mixed_x = lam * all_x + (1 - lam) * all_x[index]
        
        # Forward pass
        logits = self.predict(mixed_x)
        
        # Mixup loss
        loss = lam * F.cross_entropy(logits, all_y) + \
               (1 - lam) * F.cross_entropy(logits, all_y[index])
        
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

class GroupDRO(Algorithm):
    """Group Distributionally Robust Optimization.
    
    This algorithm minimizes the worst-case loss across domains by
    adaptively reweighting the loss for each domain.
    """
    def __init__(self, network: nn.Module, hparams: dict, device: torch.device):
        super().__init__(network, hparams, device)
        self.group_weights = None
        self.n_groups = None

    def initialize_group_weights(self, n_groups: int):
        """Initialize group weights for DRO.
        
        Args:
            n_groups (int): Number of domains/groups
        """
        self.n_groups = n_groups
        self.group_weights = torch.ones(n_groups).to(self.device) / n_groups

    def update(self, minibatches: list) -> dict:
        """Update model parameters using GroupDRO.
        
        Args:
            minibatches (list): List of (x, y) tuples for each domain
            
        Returns:
            dict: Dictionary containing loss values
        """
        if self.group_weights is None:
            self.initialize_group_weights(len(minibatches))
        
        self.optimizer.zero_grad()
        
        # Compute loss for each domain
        group_losses = []
        for x, y in minibatches:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.predict(x)
            group_losses.append(F.cross_entropy(logits, y))
        
        group_losses = torch.stack(group_losses)
        
        # Update group weights
        self.group_weights = self.group_weights * torch.exp(self.hparams['dro_eta'] * group_losses)
        self.group_weights = self.group_weights / self.group_weights.sum()
        
        # Compute weighted loss
        loss = (group_losses * self.group_weights).sum()
        
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item(), 'group_weights': self.group_weights.cpu().numpy()}

class CORAL(Algorithm):
    """Correlation Alignment.
    
    This algorithm aligns the second-order statistics (covariance) of
    features across domains to reduce domain shift.
    """
    def update(self, minibatches: list) -> dict:
        """Update model parameters using CORAL.
        
        Args:
            minibatches (list): List of (x, y) tuples for each domain
            
        Returns:
            dict: Dictionary containing loss values
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        all_x = all_x.to(self.device)
        all_y = all_y.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Get features and logits
        features = self.network.get_features(all_x)
        logits = self.network.classif(features)
        
        # Classification loss
        loss = F.cross_entropy(logits, all_y)
        
        # CORAL loss
        coral_loss = 0
        for i, j in combinations(range(len(minibatches)), 2):
            x_i = minibatches[i][0].to(self.device)
            x_j = minibatches[j][0].to(self.device)
            
            f_i = self.network.get_features(x_i)
            f_j = self.network.get_features(x_j)
            
            coral_loss += self.coral_loss(f_i, f_j)
        
        loss = loss + self.hparams['coral_lambda'] * coral_loss
        
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item(), 'coral_loss': coral_loss.item()}

    def coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CORAL loss between source and target features.
        
        Args:
            source (torch.Tensor): Source domain features
            target (torch.Tensor): Target domain features
            
        Returns:
            torch.Tensor: CORAL loss value
        """
        d = source.size(1)
        
        # Source covariance
        source = source - source.mean(0, keepdim=True)
        source_cov = torch.mm(source.t(), source) / (source.size(0) - 1)
        
        # Target covariance
        target = target - target.mean(0, keepdim=True)
        target_cov = torch.mm(target.t(), target) / (target.size(0) - 1)
        
        # Frobenius norm
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * d * d)
        
        return loss

class MMD(Algorithm):
    """Maximum Mean Discrepancy.
    
    This algorithm minimizes the MMD between feature distributions
    across domains to reduce domain shift.
    """
    def update(self, minibatches: list) -> dict:
        """Update model parameters using MMD.
        
        Args:
            minibatches (list): List of (x, y) tuples for each domain
            
        Returns:
            dict: Dictionary containing loss values
        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        all_x = all_x.to(self.device)
        all_y = all_y.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Get features and logits
        features = self.network.get_features(all_x)
        logits = self.network.classif(features)
        
        # Classification loss
        loss = F.cross_entropy(logits, all_y)
        
        # MMD loss
        mmd_loss = 0
        for i, j in combinations(range(len(minibatches)), 2):
            x_i = minibatches[i][0].to(self.device)
            x_j = minibatches[j][0].to(self.device)
            
            f_i = self.network.get_features(x_i)
            f_j = self.network.get_features(x_j)
            
            mmd_loss += self.mmd_loss(f_i, f_j)
        
        loss = loss + self.hparams['mmd_lambda'] * mmd_loss
        
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item(), 'mmd_loss': mmd_loss.item()}

    def mmd_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MMD loss between source and target features.
        
        Args:
            source (torch.Tensor): Source domain features
            target (torch.Tensor): Target domain features
            
        Returns:
            torch.Tensor: MMD loss value
        """
        kernels = [1, 2, 4, 8, 16]
        loss = 0
        
        for kernel in kernels:
            source_kernel = torch.mm(source, source.t())
            target_kernel = torch.mm(target, target.t())
            cross_kernel = torch.mm(source, target.t())
            
            loss += torch.mean(source_kernel) + torch.mean(target_kernel) - 2 * torch.mean(cross_kernel)
        
        return loss
