import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

import torch
from data_loader import CustomMEGDataset
from models import build_and_train_network

# --- Configuration ---


@dataclass
class Config:
    # config for the training pipeline
    # Data paths
    data_path: str = "data"
    output_dir: str = os.path.join("..", "models")
    # --- Data loader parameters ---
    data_root: str = data_path
    scenario: str = "intra"
    mode: str = "train"
    target_subject_id: Optional[str] = "105923"

    # --- Hyperparameters ---
    epochs: int = 15

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
# --- Device selection ---


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

# --- Main training pipeline ---


def train_pipeline(config: Config):
    # 1. Dataset and DataLoader
    dataset = CustomMEGDataset(
        data_root=config.data_root,
        scenario=config.scenario,
        mode=config.mode,
        task_to_label_map=config.task_to_label_map,
        target_subject_id=config.target_subject_id
    )

    logger.info(f"loaded dataset with shape: {dataset.data.shape}")

    # 2. Train Model
    logger.info("Building and training the model...")
    model = build_and_train_network(
        dataset,
        data_path=config.data_root,
        n_outputs=len(config.task_to_label_map),
        net_option="meegnet",
        name="taskclf_meegnet",
        max_epoch=config.epochs,
        verbose=1
    )

    # 3. Save the trained model
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        logger.info(f"Created output directory: {config.output_dir}")

    save_path = os.path.join(config.output_dir, "taskclf_meegnet.pth")
    if hasattr(model, "save"):
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    else:
        logger.warning(
            "Model could not be saved: no save or state_dict method found.")

    model.plot_loss()
    model.plot_accuracy()


# --- Entry point ---


if __name__ == "__main__":
    config = Config()
    config.device = get_device()
    train_pipeline(config)
