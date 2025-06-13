"""MEG Classification Model Evaluation

This script evaluates trained MEG classification models on test datasets.
It supports both intra-subject and cross-subject evaluation scenarios,
generating comprehensive performance reports and visualizations.

Key features:
- Model loading and evaluation
- Performance metrics calculation
- Confusion matrix visualization
- Detailed evaluation reports
"""

import os
import logging
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import optuna
from data_loader import CustomMEGDataset
from models import create_net
from algorithms import get_algorithm_class
from train import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Task mapping
TASK_MAP = {"rest": 0, "task_motor": 1, "task_story_math": 2, "task_working_memory": 3}
CLASS_NAMES = list(TASK_MAP.keys())

def load_model_for_evaluation(scenario: str, output_dir: str, data_root: str):
    """Load the best model and its configuration for evaluation.
    
    Args:
        scenario (str): 'intra' or 'cross' scenario
        output_dir (str): Directory containing model files
        data_root (str): Root directory containing data
        
    Returns:
        tuple: (algorithm, batch_size, hparams, net_option) or (None, None, None, None) if loading fails
    """
    logger.info("--- Loading best model configuration from Optuna study ---")
    study_name = f"meg-{scenario}-study-v1"
    storage_name = f"sqlite:///{study_name}.db"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        logger.error(f"Could not find study '{study_name}'. Make sure 'train.py' has been run.")
        return None, None, None, None

    best_trial = study.best_trial
    hparams = best_trial.params.copy()
    net_option = hparams.pop('net_architecture')
    algorithm_name = hparams.pop('algorithm')
    batch_size = hparams.pop('batch_size')

    if algorithm_name == 'IRM':
        hparams['irm_penalty_anneal_iters'] = 50

    dummy_dataset_config = {
        "sfreq": 500, "orig_sfreq": 2034, "window": 2, "overlap": 0.5,
        "scaling": "minmax", "random_state": 42, "lso": False,
        "split_sizes": (1.0, 0.0, 0.0)
    }
    temp_dataset = CustomMEGDataset(
        data_root=data_root,
        scenario=scenario,
        mode="train",
        task_to_label_map=TASK_MAP,
        **dummy_dataset_config
    )
    sample_input, _ = temp_dataset[0]
    input_shape, n_outputs = sample_input.shape, len(TASK_MAP)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = create_net(net_option, input_shape, n_outputs, hparams).to(device)
    model_path = os.path.join(output_dir, "final_model.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Final model not found at {model_path}. Please run train.py first.")
        return None, None, None, None

    network.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Successfully loaded model weights from {model_path}")
    
    algorithm = get_algorithm_class(algorithm_name)(network, hparams, device)
    return algorithm, batch_size, hparams, net_option

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): Names of classes
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    """Main evaluation function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    output_dir = f"output/{args.scenario}"
    os.makedirs(output_dir, exist_ok=True)
    
    algorithm, batch_size, hparams, net_option = load_model_for_evaluation(args.scenario, output_dir, args.data_root)
    if not algorithm:
        return

    DATASET_CONFIG = {
        "sfreq": 500, "orig_sfreq": 2034, "window": 2, "overlap": 0.5,
        "scaling": "minmax", "random_state": 42, "lso": (args.scenario == "cross"),
        "split_sizes": (0.0, 0.0, 1.0)
    }
    
    if args.scenario == "intra":
        test_datasets = [CustomMEGDataset(
            data_root=args.data_root,
            scenario="intra",
            mode="test",
            task_to_label_map=TASK_MAP,
            **DATASET_CONFIG
        )]
        test_names = ["Intra-Test"]
    else:
        test_datasets = [
            CustomMEGDataset(
                data_root=args.data_root,
                scenario="cross",
                mode=f"test{i}",
                task_to_label_map=TASK_MAP,
                **DATASET_CONFIG
            ) for i in [1, 2, 3]
        ]
        test_names = ["Cross-Test-1", "Cross-Test-2", "Cross-Test-3"]

    trainer = ModelTrainer(algorithm)
    full_report = f"Evaluation Report for Scenario: {args.scenario.upper()}\n"
    full_report += f"Model: {net_option}, Algorithm: {type(algorithm).__name__}\n\n"

    for name, dataset in zip(test_names, test_datasets):
        if len(dataset) == 0:
            logger.warning(f"Test set '{name}' is empty. Skipping.")
            continue
        
        logger.info(f"\n--- Evaluating on: {name} ---")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        loss, acc, y_pred, y_true = trainer.evaluate(loader)
        
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
        logger.info(f"Results for {name}:\nLoss={loss:.4f}, Accuracy={acc:.4f}\n{report}")
        
        full_report += f"--- Results for {name} ---\n"
        full_report += f"Overall Accuracy: {acc:.4f}\n"
        full_report += report + "\n" + "="*50 + "\n"
        
        cm_path = os.path.join(output_dir, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, cm_path)

    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(full_report)
    logger.info(f"\nFull evaluation report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MEG classification model.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="intra",
        choices=["intra", "cross"],
        help="The evaluation scenario ('intra' or 'cross')."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/daan/DL2/data",
        help="The root directory where the data is stored."
    )
    args = parser.parse_args()
    main(args)
