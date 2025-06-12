import os
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
import numpy as np

import optuna
from optuna.trial import TrialState

from data_loader import CustomMEGDataset
from models import create_net
from algorithms import get_algorithm_class, Algorithm, Mixup, IRM

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#                                 Global Config                                #
# ---------------------------------------------------------------------------- #
SCENARIO = "cross" 
DATA_ROOT = "/home/daan/DL2/data"
TASK_MAP = {"rest": 0, "task_motor": 1, "task_story_math": 2, "task_working_memory": 3}
DATASET_CONFIG = {
    "sfreq": 500, "orig_sfreq": 2034, "window": 2, "overlap": 0.5,
    "scaling": "minmax", "random_state": 42, "lso": (SCENARIO == "cross"),
    "split_sizes": (0.8, 0.2, 0.0) if SCENARIO == "intra" else (0.5, 0.5, 0.0)
}

# ---------------------------------------------------------------------------- #
#                           Domain-Aware Sampler                             #
# ---------------------------------------------------------------------------- #
class DomainSampler(Sampler):
    """Samples minibatches with samples from `n_domains_per_batch` domains."""
    def __init__(self, data_subset: Subset, batch_size: int, n_domains_per_batch: int):
        self.data_subset = data_subset
        self.batch_size = batch_size
        self.n_domains_per_batch = n_domains_per_batch
        all_groups = self.data_subset.dataset.groups.numpy()
        self.subset_groups = all_groups[self.data_subset.indices]
        self.unique_groups_in_subset = np.unique(self.subset_groups)
        self.group_to_subset_indices = {g: np.where(self.subset_groups == g)[0] for g in self.unique_groups_in_subset}
        
    def __iter__(self):
        n_batches = len(self.data_subset) // self.batch_size
        domain_keys = list(self.group_to_subset_indices.keys())
        for _ in range(n_batches):
            batch_indices = []
            if len(domain_keys) < self.n_domains_per_batch: continue
            chosen_domains = np.random.choice(domain_keys, self.n_domains_per_batch, replace=False)
            samples_per_domain = self.batch_size // self.n_domains_per_batch
            for domain in chosen_domains:
                subset_indices_for_domain = self.group_to_subset_indices[domain]
                if len(subset_indices_for_domain) > 0:
                    chosen_local_indices = np.random.choice(subset_indices_for_domain, samples_per_domain, replace=True)
                    batch_indices.extend(chosen_local_indices)
            yield batch_indices

    def __len__(self):
        return len(self.data_subset) // self.batch_size

# ---------------------------------------------------------------------------- #
#                               Training Wrapper                               #
# ---------------------------------------------------------------------------- #
class ModelTrainer:
    """A wrapper to handle the training and evaluation loop."""
    def __init__(self, algorithm: Algorithm, trial: Optional[optuna.Trial] = None):
        self.algorithm = algorithm
        self.device = algorithm.device
        self.trial = trial
        self.save_path = f"output/{SCENARIO}/trial_{trial.number}_model.pt" if trial else "output/best_model.pt"
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.scheduler = StepLR(self.algorithm.optimizer, step_size=10, gamma=0.5)

    def evaluate(self, dataloader):
        self.algorithm.network.eval()
        correct, total, losses = 0, 0, 0.0
        if not dataloader: return 0.0, 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.algorithm.predict(x)
                losses += F.cross_entropy(logits, y, reduction='sum').item()
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        self.algorithm.network.train()
        if total == 0: return 0.0, 0.0
        return losses / total, correct / total

    def train(self, dataset, batch_size, max_epochs, patience):
        train_indices, valid_indices, _ = dataset.split_data()

        if not valid_indices:
            logger.warning("Validation set is empty. Pruning trial.")
            raise optuna.exceptions.TrialPruned("Empty validation set.")
            
        train_subset = Subset(dataset, train_indices)
        valid_subset = Subset(dataset, valid_indices)
        
        if SCENARIO == 'cross' and isinstance(self.algorithm, (IRM, Mixup)):
            logger.info(f"Using DomainSampler for {type(self.algorithm).__name__}")
            unique_train_groups = np.unique(train_subset.dataset.groups[train_subset.indices])
            if len(unique_train_groups) < 2:
                 train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            else:
                train_sampler = DomainSampler(train_subset, batch_size, n_domains_per_batch=2)
                train_loader = DataLoader(train_subset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        for epoch in range(max_epochs):
            self.algorithm.network.train()
            
            # The algorithm's .update() method now handles the optimizer step and zero_grad.
            if isinstance(self.algorithm, Mixup):
                for (x1, y1), (x2, y2) in zip(train_loader, train_loader):
                    self.algorithm.update([(x1, y1), (x2, y2)])
            else:
                for x, y in train_loader:
                    self.algorithm.update([(x.to(self.device), y.to(self.device))])
            
            train_loss, train_acc = self.evaluate(train_loader)
            valid_loss, valid_acc = self.evaluate(valid_loader)
            self.scheduler.step()
            
            logger.info(f"E{epoch+1} | VAcc: {valid_acc:.4f} | VLoss: {valid_loss:.4f} | TAcc: {train_acc:.4f} | LR: {self.scheduler.get_last_lr()[0]:.1E}")
            
            if valid_acc > self.best_val_acc:
                self.best_val_acc = valid_acc
                self.patience_counter = 0
                if self.save_path:
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    torch.save(self.algorithm.network.state_dict(), self.save_path)
            else:
                self.patience_counter += 1

            if self.trial:
                self.trial.report(valid_acc, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break
        
        return self.best_val_acc

# ---------------------------------------------------------------------------- #
#                                 Optuna & Main                                #
# ---------------------------------------------------------------------------- #
def objective(trial: optuna.Trial) -> float:
    try:
        hparams = {
            "lr": trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float('dropout', 0.1, 0.5),
            "n_filters_1": trial.suggest_categorical('n_filters_1', [8, 16, 32]),
            "n_filters_2": trial.suggest_categorical('n_filters_2', [16, 32, 64]),
            "kernel_length": trial.suggest_categorical('kernel_length', [32, 64]),
            "n_layers": trial.suggest_int('n_layers', 1, 3),
            "n_head": trial.suggest_categorical('n_head', [2, 4]),
            "d_model": trial.suggest_categorical('d_model', [64, 128]),
        }
        net_option = trial.suggest_categorical('net_architecture', ['meegnet', 'eegnet', 'attention'])
        algorithm_name = trial.suggest_categorical('algorithm', ['ERM', 'IRM', 'Mixup'])
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        if algorithm_name == 'IRM':
            hparams['irm_lambda'] = trial.suggest_float('irm_lambda', 0.1, 10.0, log=True)
            hparams['irm_penalty_anneal_iters'] = 50
        elif algorithm_name == 'Mixup':
            hparams['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.1, 0.5)

        train_val_dataset = CustomMEGDataset(data_root=DATA_ROOT, scenario=SCENARIO, mode="train", task_to_label_map=TASK_MAP, **DATASET_CONFIG)
        if len(train_val_dataset) == 0: raise optuna.exceptions.TrialPruned("Dataset is empty.")

        sample_input, _ = train_val_dataset[0]
        input_shape, n_outputs = sample_input.shape, 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        network = create_net(net_option, input_shape, n_outputs, hparams).to(device)
        algorithm = get_algorithm_class(algorithm_name)(network, hparams, device)
        
        logger.info(f"TRIAL {trial.number}: {net_option}/{algorithm_name}, BS={batch_size}, LR={hparams['lr']:.1E}")
        trainer = ModelTrainer(algorithm, trial)
        best_val_acc = trainer.train(train_val_dataset, batch_size, max_epochs=50, patience=7)

        return best_val_acc
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return 0.0

def test_model(best_trial: optuna.trial.FrozenTrial):
    logger.info("\n--- TESTING BEST MODEL ---")
    hparams = best_trial.params.copy()
    net_option = hparams.pop('net_architecture')
    algorithm_name = hparams.pop('algorithm')
    batch_size = hparams.pop('batch_size')
    
    if SCENARIO == "intra":
        test_datasets = [CustomMEGDataset(data_root=DATA_ROOT, scenario="intra", mode="test", task_to_label_map=TASK_MAP, **DATASET_CONFIG)]
        test_names = ["Intra-Test"]
    else:
        test_datasets = [CustomMEGDataset(data_root=DATA_ROOT, scenario="cross", mode=f"test{i}", task_to_label_map=TASK_MAP, **DATASET_CONFIG) for i in [1, 2, 3]]
        test_names = ["Cross-Test1", "Cross-Test2", "Cross-Test3"]

    sample_input, _ = test_datasets[0][0]
    input_shape, n_outputs = sample_input.shape, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = create_net(net_option, input_shape, n_outputs, hparams).to(device)
    model_path = f"output/{SCENARIO}/trial_{best_trial.number}_model.pt"
    if os.path.exists(model_path):
        network.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded best model weights from {model_path}")
    else:
        logger.error(f"Could not find model weights at {model_path}. Cannot perform testing.")
        return

    algorithm = get_algorithm_class(algorithm_name)(network, hparams, device)
    trainer = ModelTrainer(algorithm)

    for name, dataset in zip(test_names, test_datasets):
        if len(dataset) > 0:
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            test_loss, test_acc = trainer.evaluate(test_loader)
            logger.info(f"Results for {name}: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")
        else:
            logger.warning(f"Test set {name} is empty. Skipping.")

def main():
    if not os.path.exists(DATA_ROOT):
        logger.error(f"Data directory '{DATA_ROOT}' not found. Please update the path.")
        return
        
    study_name = f"meg-{SCENARIO}-study-v1"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True,
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=100)
    
    logger.info(f"\n--- OPTIMIZATION FINISHED FOR '{SCENARIO}' SCENARIO ---")
    logger.info(f"Best trial: {study.best_trial.number} with value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    test_model(study.best_trial)

if __name__ == "__main__":
    main()
