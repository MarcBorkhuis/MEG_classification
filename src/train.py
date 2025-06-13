"""MEG Classification Model Training

This script handles the training and hyperparameter optimization of MEG classification models.
It supports both intra-subject and cross-subject scenarios, with various domain adaptation
algorithms and neural network architectures.

Key features:
- Hyperparameter optimization using Optuna
- Support for multiple training algorithms (ERM, IRM, Mixup, etc.)
- Domain adaptation for cross-subject scenarios
- Early stopping and model checkpointing
"""

import os
import logging
from typing import Optional
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler, ConcatDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
import optuna

from data_loader import CustomMEGDataset
from models import create_net
# Import all algorithms AND the ALGORITHMS list from the revised script
from algorithms import get_algorithm_class, ALGORITHMS, Algorithm, Mixup, IRM, GroupDRO, CORAL, MMD

# --- Basic Setup ---
# Set level to DEBUG to see all log messages
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#                                 Global Config                                #
# ---------------------------------------------------------------------------- #
# Set this to 'intra' or 'cross' to control the training execution
SCENARIO = "cross" 
# Make sure this path is correct for your system
DATA_ROOT = "/home/daan/DL2/data" 
# UPDATED: Changed output directory to v2
OUTPUT_DIR = f"outputv2/{SCENARIO}"
TASK_MAP = {"rest": 0, "task_motor": 1, "task_story_math": 2, "task_working_memory": 3}
# Default config, will be adapted based on scenario
DATASET_CONFIG = {
    "sfreq": 500, "orig_sfreq": 2034, "window": 2, "overlap": 0.5,
    "scaling": "minmax", "random_state": 42, "lso": (SCENARIO == "cross"),
    "split_sizes": (0.8, 0.2, 0.0) 
}

# ---------------------------------------------------------------------------- #
#                         Custom Dataset and Sampler                           #
# ---------------------------------------------------------------------------- #

class CustomConcatDataset(ConcatDataset):
    """Wrapper for ConcatDataset that preserves group information."""
    def __init__(self, datasets):
        super().__init__(datasets)
        self.groups = torch.cat([d.groups for d in self.datasets])

class DomainSampler(Sampler):
    """Sampler for multi-domain batch construction."""
    def __init__(self, data_subset: Subset, samples_per_domain: int, n_domains_per_batch: int):
        self.data_subset = data_subset
        self.samples_per_domain = samples_per_domain
        self.n_domains_per_batch = n_domains_per_batch
        all_groups = self.data_subset.dataset.groups.numpy()
        subset_groups = all_groups[self.data_subset.indices]
        self.unique_groups_in_subset = np.unique(subset_groups)
        self.group_to_subset_indices = {g: np.where(subset_groups == g)[0] for g in self.unique_groups_in_subset}
        logger.debug(f"DomainSampler initialized for {len(self.unique_groups_in_subset)} domains.")

        if not self.group_to_subset_indices or len(self.unique_groups_in_subset) < self.n_domains_per_batch:
            self.num_batches = 0
            logger.warning("Not enough domains in dataset for DomainSampler, will yield 0 batches.")
        else:
            min_domain_size = min(len(indices) for indices in self.group_to_subset_indices.values())
            self.num_batches = min_domain_size // self.samples_per_domain
        logger.debug(f"DomainSampler will yield {self.num_batches} batches per epoch.")

    def __iter__(self):
        domain_keys = list(self.group_to_subset_indices.keys())
        for _ in range(self.num_batches):
            flat_batch_indices = []
            chosen_domains = np.random.choice(domain_keys, self.n_domains_per_batch, replace=False)
            for domain in chosen_domains:
                subset_indices_for_domain = self.group_to_subset_indices[domain]
                chosen_local_indices = np.random.choice(subset_indices_for_domain, self.samples_per_domain, replace=True)
                flat_batch_indices.extend(chosen_local_indices.tolist())
            yield flat_batch_indices

    def __len__(self):
        return self.num_batches

# ---------------------------------------------------------------------------- #
#                               Training Wrapper                               #
# ---------------------------------------------------------------------------- #
class ModelTrainer:
    """Training and evaluation wrapper for MEG classification models."""
    def __init__(self, algorithm: Algorithm, trial: Optional[optuna.Trial] = None, final_training=False):
        self.algorithm = algorithm
        self.device = algorithm.device
        self.trial = trial
        self.final_training = final_training
        self.save_path = os.path.join(OUTPUT_DIR, f"trial_{trial.number}_model.pt") if trial else os.path.join(OUTPUT_DIR, "final_model.pt")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.scheduler = StepLR(self.algorithm.optimizer, step_size=10, gamma=0.5)
        self.training_log = []

    def evaluate(self, dataloader):
        """Evaluate model on a dataset."""
        self.algorithm.network.eval()
        correct, total, losses = 0, 0, 0.0
        if not dataloader: return 0.0, 0.0, [], []
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.algorithm.predict(x)
                losses += F.cross_entropy(logits, y, reduction='sum').item()
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        self.algorithm.network.train()
        return (losses / total, correct / total, all_preds, all_labels) if total > 0 else (0.0, 0.0, [], [])

    def train(self, train_dataset, batch_size, max_epochs, patience, valid_dataset=None):
        """Train the model."""
        if self.final_training:
            logger.info(f"Final training mode. Using all {len(train_dataset)} samples for training.")
            train_subset = Subset(train_dataset, range(len(train_dataset)))
            valid_loader = None
        elif valid_dataset:
            logger.info(f"Using provided train dataset ({len(train_dataset)} samples) and validation dataset ({len(valid_dataset)} samples).")
            train_subset = Subset(train_dataset, range(len(train_dataset)))
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            logger.info(f"Splitting dataset of {len(train_dataset)} samples for training and validation.")
            train_indices, valid_indices, _ = train_dataset.split_data()
            if not valid_indices:
                raise optuna.exceptions.TrialPruned("Empty validation set after splitting.")
            
            train_subset = Subset(train_dataset, train_indices)
            valid_loader = DataLoader(Subset(train_dataset, valid_indices), batch_size=batch_size, shuffle=False, num_workers=4)
            logger.info(f"Train subset size: {len(train_subset)}, Validation subset size: {len(valid_loader.dataset)}")

        MULTI_DOMAIN_ALGORITHMS = ["IRM", "Mixup", "GroupDRO", "CORAL", "MMD"]
        use_domain_sampler = SCENARIO == 'cross' and type(self.algorithm).__name__ in MULTI_DOMAIN_ALGORITHMS

        if use_domain_sampler:
            logger.info(f"Using DomainSampler for {type(self.algorithm).__name__}")
            n_domains_per_batch = 2
            samples_per_domain = batch_size // n_domains_per_batch
            
            train_sampler = DomainSampler(train_subset, samples_per_domain=samples_per_domain, n_domains_per_batch=n_domains_per_batch)
            train_loader = DataLoader(train_subset, batch_sampler=train_sampler, num_workers=4)
            
            if isinstance(self.algorithm, GroupDRO):
                self.algorithm.initialize_group_weights(train_sampler.n_domains_per_batch)
                logger.debug("Initialized GroupDRO weights.")
        else:
            logger.info("Using standard DataLoader.")
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        for epoch in range(max_epochs):
            self.algorithm.network.train()
            epoch_losses = []
            
            for i, data in enumerate(train_loader):
                if use_domain_sampler:
                    x, y = data
                    x_s = torch.chunk(x, chunks=train_sampler.n_domains_per_batch, dim=0)
                    y_s = torch.chunk(y, chunks=train_sampler.n_domains_per_batch, dim=0)
                    minibatches = list(zip(x_s, y_s))
                else:
                    minibatches = [data]

                loss_dict = self.algorithm.update([(x.to(self.device), y.to(self.device)) for x, y in minibatches])
                epoch_losses.append(loss_dict)
            
            if not epoch_losses:
                logger.warning(f"Epoch {epoch+1} had no batches to process. Skipping logging for this epoch.")
                continue
                
            avg_losses = {k: np.mean([d[k] for d in epoch_losses if k in d]) for k in epoch_losses[0]}
            log_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            
            if self.final_training:
                logger.info(f"E{epoch+1}/{max_epochs} | {log_str}")
            else:
                valid_loss, valid_acc, _, _ = self.evaluate(valid_loader)
                logger.info(f"E{epoch+1} | VAcc: {valid_acc:.4f} | VLoss: {valid_loss:.4f} | {log_str}")

                if valid_acc > self.best_val_acc:
                    self.best_val_acc, self.patience_counter = valid_acc, 0
                    torch.save(self.algorithm.network.state_dict(), self.save_path)
                    logger.debug(f"New best validation accuracy: {valid_acc:.4f}. Model saved to {self.save_path}")
                else:
                    self.patience_counter += 1

                if self.trial:
                    self.trial.report(valid_acc, epoch)
                    if self.trial.should_prune(): raise optuna.exceptions.TrialPruned()
                
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} due to no improvement.")
                    break
            self.training_log.append(avg_losses)
            self.scheduler.step()

        if self.final_training: torch.save(self.algorithm.network.state_dict(), self.save_path)
        return self.best_val_acc

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""
    try:
        # FIX: Define all possible hyperparameters for all trials to ensure a static search space
        hparams = {
            # General
            "lr": trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float('dropout', 0.1, 0.5),
            
            # EEGNet
            "n_filters_1": trial.suggest_categorical('eegnet_n_filters_1', [8, 16, 32]),
            "depth_multiplier": trial.suggest_categorical('eegnet_depth_multiplier', [2, 4]),
            "kernel_length": trial.suggest_categorical('eegnet_kernel_length', [32, 64]),

            # MEEGNet
            "meegnet_n_filters_1": trial.suggest_categorical('meegnet_n_filters_1', [16, 32, 64]),
            "meegnet_n_filters_2": trial.suggest_categorical('meegnet_n_filters_2', [32, 64, 128]),

            # Attention
            "n_layers": trial.suggest_int('attn_n_layers', 1, 3),
            "n_head": trial.suggest_categorical('attn_n_head', [2, 4]),
            "d_model": trial.suggest_categorical('attn_d_model', [64, 128]),
            
            # Algorithms
            'irm_lambda': trial.suggest_float('irm_lambda', 0.1, 10.0, log=True),
            'irm_penalty_anneal_iters': 50,
            'mixup_alpha': trial.suggest_float('mixup_alpha', 0.1, 0.5),
            'dro_eta': trial.suggest_float('dro_eta', 1e-2, 1e-1, log=True),
            'coral_lambda': trial.suggest_float('coral_lambda', 0.1, 10.0, log=True),
            'mmd_lambda': trial.suggest_float('mmd_lambda', 0.1, 10.0, log=True)
        }
        
        net_option = trial.suggest_categorical('net_architecture', ['meegnet', 'eegnet', 'attention'])
        algorithm_name = trial.suggest_categorical('algorithm', ALGORITHMS)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        if SCENARIO == 'cross':
            logger.info("Cross-scenario HPO: Training on all 'train' data, validating on 'test1'.")
            train_dataset = CustomMEGDataset(data_root=DATA_ROOT, scenario='cross', mode="train", task_to_label_map=TASK_MAP, **DATASET_CONFIG)
            valid_dataset = CustomMEGDataset(data_root=DATA_ROOT, scenario='cross', mode="test1", task_to_label_map=TASK_MAP, **DATASET_CONFIG)
            if len(train_dataset) == 0 or len(valid_dataset) == 0:
                raise optuna.exceptions.TrialPruned("Train or validation data is empty.")
            trainer_args = (train_dataset, batch_size, 50, 7, valid_dataset)
        else:
            logger.info("Intra-scenario HPO: Splitting 'train' data for training and validation.")
            train_val_dataset = CustomMEGDataset(data_root=DATA_ROOT, scenario='intra', mode="train", task_to_label_map=TASK_MAP, **DATASET_CONFIG)
            if len(train_val_dataset) == 0:
                raise optuna.exceptions.TrialPruned("Dataset is empty.")
            trainer_args = (train_val_dataset, batch_size, 50, 7)

        sample_input_dataset = train_dataset if SCENARIO == 'cross' else train_val_dataset
        sample_input, _ = sample_input_dataset[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pass the full hparams dict; models and algorithms will use .get() to find what they need
        network = create_net(net_option, sample_input.shape, len(TASK_MAP), hparams).to(device)
        algorithm = get_algorithm_class(algorithm_name)(network, hparams, device)
        
        logger.info(f"--- Starting Trial {trial.number}: {net_option}/{algorithm_name}, BS={batch_size}, LR={hparams['lr']:.1E} ---")
        trainer = ModelTrainer(algorithm, trial)
        return trainer.train(*trainer_args)

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"TRIAL {trial.number} FAILED: {e}", exc_info=True)
        return 0.0

def train_final_model(best_trial: optuna.trial.FrozenTrial):
    """Train the final model using the best hyperparameters."""
    logger.info("\n" + "="*50 + "\n--- TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS ---\n" + "="*50)
    logger.info(f"Best Trial: #{best_trial.number} with Accuracy: {best_trial.value:.4f}")
    hparams = best_trial.params.copy()
    logger.info(f"Best Hyperparameters: {hparams}")

    net_option = hparams.pop('net_architecture')
    algorithm_name = hparams.pop('algorithm')
    batch_size = hparams.pop('batch_size')
    
    if SCENARIO == 'cross':
        logger.info("Cross-scenario Final Training: Using ONLY data from the 'train' folder.")
        final_dataset = CustomMEGDataset(data_root=DATA_ROOT, scenario='cross', mode="train", task_to_label_map=TASK_MAP, **DATASET_CONFIG)
    else: 
        logger.info("Intra-scenario Final Training: Using all data from 'train' mode.")
        final_dataset = CustomMEGDataset(data_root=DATA_ROOT, scenario='intra', mode="train", task_to_label_map=TASK_MAP, **DATASET_CONFIG)
    
    logger.info(f"Final training dataset size: {len(final_dataset)} samples.")
    sample_input, _ = final_dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = create_net(net_option, sample_input.shape, len(TASK_MAP), hparams).to(device)
    algorithm = get_algorithm_class(algorithm_name)(network, hparams, device)
    
    trainer = ModelTrainer(algorithm, final_training=True)
    trainer.train(final_dataset, batch_size, max_epochs=50, patience=float('inf'))

    pd.DataFrame(trainer.training_log).to_csv(os.path.join(OUTPUT_DIR, "final_model_training_log.csv"), index=False)
    logger.info(f"Final model training complete. Model saved to {trainer.save_path}")

def main():
    """Main training function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    study_name = f"meg-{SCENARIO}-study-v2"
    storage_name = f"sqlite:///{os.path.join(OUTPUT_DIR, study_name)}.db"
    
    logger.info(f"Starting Optuna study: {study_name}")
    logger.info(f"Database will be stored at: {storage_name}")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    n_total_trials = 150
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed < n_total_trials:
        n_new_trials = n_total_trials - n_completed
        logger.info(f"Study has {n_completed} completed trials. Running {n_new_trials} more.")
        study.optimize(objective, n_trials=n_new_trials, n_jobs=1)  # n_jobs=1 for thread-safe logging
    else:
        logger.info("Study has already reached the target number of trials.")

    logger.info(f"\n--- OPTIMIZATION FINISHED FOR '{SCENARIO}' ---")
    try:
        best_trial = study.best_trial
        logger.info(f"Best trial: #{best_trial.number} with value: {best_trial.value}")
        
        df = study.trials_dataframe().sort_values("value", ascending=False)
        df.to_csv(os.path.join(OUTPUT_DIR, "optuna_study_results.csv"), index=False)
        
        train_final_model(best_trial)
    except ValueError:
        logger.error("No completed trials found in the study. Cannot determine best trial.")

    logger.info("\n--- PROCESS COMPLETE ---")
    logger.info(f"Next step: Run 'evaluate.py --scenario {SCENARIO}' to test the final model.")

if __name__ == "__main__":
    main()
