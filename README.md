# MEG Classification with Domain Adaptation

This repository contains a PyTorch implementation of MEG (Magnetoencephalography) classification with domain adaptation techniques. The project focuses on improving cross-subject generalization in MEG classification tasks by implementing various domain adaptation and regularization algorithms.

## Features

- **Multiple Neural Network Architectures**:
  - EEGNet: Lightweight CNN optimized for EEG/MEG data
  - MEEGNet: Modified CNN for MEG data
  - MEGNetMultiHead: Transformer-based architecture with multi-head attention

- **Domain Adaptation Algorithms**:
  - ERM: Empirical Risk Minimization (baseline)
  - IRM: Invariant Risk Minimization
  - Mixup: Data augmentation through interpolation
  - GroupDRO: Group Distributionally Robust Optimization
  - CORAL: Correlation Alignment
  - MMD: Maximum Mean Discrepancy

- **Training Features**:
  - Hyperparameter optimization using Optuna
  - Early stopping and model checkpointing
  - Support for both intra-subject and cross-subject scenarios
  - Comprehensive evaluation metrics and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MEG_classification.git
cd MEG_classification
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Structure

The project expects MEG data in H5 format, organized as follows:
```
data/
├── intra/
│   ├── train/
│   └── test/
└── cross/
    ├── train/
    ├── test1/
    ├── test2/
    └── test3/
```

Each H5 file should contain MEG recordings with the following naming convention:
`{task_type}_{subject_id}.h5`

## Usage

### Training

1. Configure the training scenario in `src/train.py`:
```python
SCENARIO = "cross"  # or "intra"
DATA_ROOT = "/path/to/your/data"
```

2. Run the training script:
```bash
python src/train.py
```

The script will:
- Perform hyperparameter optimization using Optuna
- Train the final model with the best hyperparameters
- Save the model and training logs

### Evaluation

Evaluate a trained model using:
```bash
python src/evaluate.py --scenario cross --data_root /path/to/your/data
```

This will:
- Load the best model for the specified scenario
- Evaluate on test datasets
- Generate performance reports and visualizations

## Project Structure

```
MEG_classification/
├── src/
│   ├── algorithms.py    # Domain adaptation algorithms
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── models.py        # Neural network architectures
│   ├── train.py         # Training and optimization
│   ├── evaluate.py      # Model evaluation
│   └── utils.py         # Utility functions
├── output/             # Training outputs and logs
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Domain Adaptation Techniques

The project implements several domain adaptation techniques specifically designed for MEG classification:

1. **CORAL (Correlation Alignment)**:
   - Aligns second-order statistics (covariance) of features across domains
   - Particularly effective for MEG data due to its focus on temporal dynamics

2. **MMD (Maximum Mean Discrepancy)**:
   - Minimizes the distance between feature distributions across domains
   - Uses multiple kernel scales to capture different aspects of the data

3. **GroupDRO (Group Distributionally Robust Optimization)**:
   - Adaptively reweights the loss for each subject/domain
   - Ensures robust performance across different subjects

## Citation

If you use this code in your research, please cite:
```
@software{meg_classification,
  author = {Your Name},
  title = {MEG Classification with Domain Adaptation},
  year = {2024},
  url = {https://github.com/yourusername/MEG_classification}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
