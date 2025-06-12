import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import copy
import numpy as np
from utils import random_pairs_of_minibatches

ALGORITHMS = ["ERM", "IRM", "Mixup", "GroupDRO", "CORAL"]

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(f"Algorithm not found: {algorithm_name}")
    return globals()[algorithm_name]

class Algorithm(nn.Module):
    """Base class for all training algorithms."""
    def __init__(self, network: nn.Module, hparams: dict, device):
        super().__init__()
        self.network = network
        self.hparams = hparams
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        raise NotImplementedError

    def predict(self, x):
        return self.network(x)

class ERM(Algorithm):
    """Empirical Risk Minimization."""
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches]).to(self.device)
        all_y = torch.cat([y for x, y in minibatches]).long().to(self.device)
        
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        # FIX: Add gradient clipping for training stability.
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, network, hparams, device):
        super().__init__(network, hparams, device)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).to(logits.device).requires_grad_()
        loss = F.cross_entropy(logits * scale, y)
        g = grad(loss, [scale], create_graph=True)[0]
        return torch.sum(g**2)

    def update(self, minibatches, unlabeled=None):
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        
        nll = 0.0
        penalty = 0.0

        for x, y in minibatches:
            x, y = x.to(self.device), y.long().to(self.device)
            logits = self.predict(x)
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        
        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        # FIX: Add gradient clipping.
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}

class Mixup(ERM):
    """Mixup of minibatches from different domains"""
    def update(self, minibatches, unlabeled=None):
        objective = torch.tensor(0.0, device=self.device)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            xi, yi = xi.to(self.device), yi.long().to(self.device)
            xj, yj = xj.to(self.device), yj.long().to(self.device)
            
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])
            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        # FIX: Add gradient clipping.
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": objective.item()}
