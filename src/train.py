import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

import torch
from data_loader import CustomMEGDataset
from models import CustomMEGModel

from torch.utils.data import ConcatDataset

# --- Configuration ---


@dataclass
class Config:
    # config for the training pipeline
    # Data paths
    # Absolute or relative path to the root data directory which contains
    data_path: str = "data"
    # 'Intra' and 'Cross' subfolders.
    output_dir: str = "models"

    # --- Data loader parameters ---
    data_root: str = data_path
    scenario: str = "intra"
    target_subject_id: Optional[str] = "105923"
    target_frequency: float = 500  # Preferred frequency in Hz

    # --- Model selection ---
    # Network architecture option ("MLP", "meegnet", "custom", "VGG", "EEGNet", "vanPutNet")
    net_option: str = "meegnet"

    # --- Hyperparameters ---
    epochs: int = 30
    batch_size: int = 1
    learning_rate: float = 0.0001

    # --- Task mapping ---
    task_to_label_map: Dict[str, int] = field(default_factory=lambda: {
        "rest": 0,
        "task_motor": 1,
        "task_story_math": 2,
        "task_working_memory": 3,
    })

    # --- Net parameters for custom model ---
    # Only use when net_option is set to 'custom'
    custom_net_params: Optional[Dict] = field(default_factory=lambda: {
        "hlayers": 3,
        "filters": 32,
        "nchan": 248,
        "linear": 100,
        "dropout": 0.5,
        "batchnorm": True,
        "maxpool": True,
    })


# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- load dataset ---


def load_dataset(config: Config):
    logger.info("Loading training dataset...")

    if config.scenario == "intra":
        train_dataset = CustomMEGDataset(
            data_root=config.data_root,
            scenario=config.scenario,
            mode="train",
            task_to_label_map=config.task_to_label_map,
            target_subject_id=config.target_subject_id,
            sfreq=config.target_frequency,
        )

        test_dataset = CustomMEGDataset(
            data_root=config.data_root,
            scenario=config.scenario,
            mode="test",
            task_to_label_map=config.task_to_label_map,
            target_subject_id=config.target_subject_id,
            sfreq=config.target_frequency,
        )

    elif config.scenario == "cross":

        modes = ["train", "test1", "test2", "test3"]
        datasets = []

        for mode in modes:
            dataset = CustomMEGDataset(
                data_root=config.data_root,
                scenario=config.scenario,
                mode=mode,
                task_to_label_map=config.task_to_label_map,
                target_subject_id=config.target_subject_id if mode == "train" else None,
                sfreq=config.target_frequency
            )
            datasets.append(dataset)

        # Concatenate all test datasets into one
        train_dataset = datasets[0]
        test_dataset = ConcatDataset(datasets[1:])

    else:
        logger.error(f"Unknown scenario: {config.scenario}")

    return train_dataset, test_dataset

# --- Main training pipeline ---


def train_pipeline(config: Config):
    # 1. Dataset and DataLoader

    train_dataset, test_dataset = load_dataset(config)
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        logger.fatal("One or both datasets are empty. Aborting training.")
        raise RuntimeError("Empty dataset encountered.")
    else:
        logger.info(
            f"Loaded training dataset with shape: {train_dataset.data.shape}")
        logger.info(
            f"Loaded test dataset with shape: {test_dataset.data.shape}")

    # 2. Train Model
    logger.info("Building model...")
    model = CustomMEGModel(
        name=f"MEGModel_{config.scenario}_{config.net_option}",
        net_option=config.net_option,
        input_size=train_dataset.data[0].shape,
        n_outputs=len(config.task_to_label_map),
        save_path=config.output_dir,
        learning_rate=config.learning_rate,


        # Optional parameters for Optuna
    )

    logger.info("Training model...")

    model.train(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        max_epoch=config.epochs,
        patience=5,
        continue_training=True,
        early_stop="loss",
    )
    logger.info("Training completed.")

    fig = model.plot_loss()
    fig.savefig(os.path.join("./", "loss.png"))
    fig = model.plot_accuracy()
    fig.savefig(os.path.join("./", "accuracy.png"))

    for key, value in model.tracker.best.items():
        logger.info(f"{key}, {value}")

    # 3. Test Model
    logger.info("Testing model...")

    model.test(
        dataset=test_dataset
    )
    logger.info("Testing completed.")


# --- Entry point ---


if __name__ == "__main__":
    config = Config()
    train_pipeline(config)
