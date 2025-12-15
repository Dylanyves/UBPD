# UBPD: Ultrasound Brachial Plexus Detection

## Overview

UBPD is a deep learning project for training and evaluating U-Net and U-Net++ models to perform semantic segmentation of ultrasound images of the brachial plexus. The project focuses on automatically detecting and segmenting anatomical structures (arteries, veins, muscles, and nerves) within ultrasound images using convolutional neural networks.

### Core Functionality

The project provides two main functionalities:

1. **Training**: Implements a full cross-validation training pipeline with support for multiple model architectures (U-Net and U-Net++). Features include:
   - Multi-fold cross-validation with customizable class subsets
   - Mixed precision (FP16) training for improved performance
   - Early stopping and best-weight restoration
   - Wandb integration for experiment tracking
   - Support for both binary and multi-class segmentation

2. **Evaluation**: Comprehensive model evaluation on held-out test sets with:
   - Per-fold or single model checkpoint evaluation
   - Multiple segmentation metrics (Dice coefficient, IoU, etc.)
   - Visualization and metric aggregation
   - Flexible checkpoint discovery and loading

---
a# UBPD: Ultrasound Brachial Plexus Detection

## Overview

UBPD is a deep learning project for training and evaluating U-Net and U-Net++ models to perform semantic segmentation of ultrasound images of the brachial plexus. The project focuses on automatically detecting and segmenting anatomical structures (arteries, veins, muscles, and nerves) within ultrasound images using convolutional neural networks.

### Core Functionality

The project provides two main functionalities:

1. **Training**: Implements a full cross-validation training pipeline with support for multiple model architectures (U-Net and U-Net++). Features include:
   - Multi-fold cross-validation with customizable class subsets
   - Mixed precision (FP16) training for improved performance
   - Early stopping and best-weight restoration
   - Wandb integration for experiment tracking
   - Support for both binary and multi-class segmentation

2. **Evaluation**: Comprehensive model evaluation on held-out test sets with:
   - Per-fold or single model checkpoint evaluation
   - Multiple segmentation metrics (Dice coefficient, IoU, etc.)
   - Visualization and metric aggregation
   - Flexible checkpoint discovery and loading

---

## Installation Guide

### Option 1: Using `uv` (Recommended)

`uv` is a fast, reliable Python package installer that improves upon pip and poetry. It's the recommended approach for this project.

#### 1.1 Install `uv` check documentation [here](https://docs.astral.sh/uv/getting-started/installation/)

For Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or install via your package manager:
```bash
# Homebrew (macOS)
brew install uv

# Cargo (if you have Rust)
cargo install uv
```

#### 1.2 Clone the Repository

```bash
git clone https://github.com/Dylanyves/UBPD.git
cd UBPD
```

#### 1.3 Install Dependencies

```bash
uv sync
```

This command will create a virtual environment and install all dependencies specified in `pyproject.toml`.

### Option 2: Using `pipenv`

If you prefer pipenv:

```bash
# Install pipenv if not already installed
pip install pipenv

# Clone and navigate to the project
git clone https://github.com/Dylanyves/UBPD.git
cd UBPD

# Install dependencies
pipenv install
pipenv shell  # Activate the virtual environment
```

---

## Usage Examples

### Training a Model

To train a U-Net model with cross-validation on all anatomical classes:

```bash
uv run main.py \
  --model unet \
  --epochs 100 \
  --batch_size 16 \
  --cv 5 \
  --include_classes 1 2 3 4 \
  --seed 42
```

To train a U-Net++ model focusing only on nerve and artery detection:

```bash
uv run main.py \
  --model unetpp \
  --epochs 150 \
  --batch_size 32 \
  --cv 5 \
  --include_classes 1 4 \
  --seed 42 \
  --use_wandb
```

Key parameters:
- `--model`: Model architecture (`unet` or `unetpp`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--cv`: Number of cross-validation folds
- `--include_classes`: Class IDs to include (1=artery, 2=vein, 3=muscle, 4=nerve)
- `--seed`: Random seed for reproducibility
- `--use_wandb`: Enable Weights & Biases tracking

### Evaluating a Model

To evaluate a trained model on the test set:

```bash
# Evaluate a 5-fold CV model
uv run evaluate.py \
  --model_id 616483 \
  --model_name unet \
  --cv \
  --include_classes 1 2 3 4
```

To evaluate a single model and visualize predictions:

```bash
# Evaluate with visualization
uv run evaluate.py \
  --model_id 616483 \
  --model_name unetpp \
  --include_classes 1 4 \
  --show_plot
```

Key parameters:
- `--model_id`: Model/experiment ID (matches checkpoint filename)
- `--model_name`: Model architecture used (`unet` or `unetpp`)
- `--cv`: Flag to evaluate multi-fold model
- `--include_classes`: Classes included in training
- `--show_plot`: Display prediction visualizations

---

## Project Structure

### Core Modules

```
src/
├── models/
│   ├── unet.py           # U-Net architecture implementation
│   └── unetpp.py         # U-Net++ architecture implementation
├── train.py              # Main Trainer class for training loop
├── evaluate.py           # Evaluator class for model evaluation
├── dataset.py            # UBPDataset class for data loading
├── train_utils.py        # Training utilities (loss, optimizer, metrics)
├── helper.py             # Helper functions (model building, transforms)
├── const.py              # Constants and paths
└── preprocessing.py      # Data preprocessing utilities
```

### Key Files

- **`main.py`**: Entry point for training. Orchestrates:
  - Cross-validation fold splitting
  - Dataset creation with class filtering and remapping
  - Training pipeline with Wandb logging
  - Test set evaluation and metric aggregation

- **`evaluate.py`**: Standalone evaluation CLI. Loads pre-trained checkpoints and evaluates on test set with metrics computation and visualization support

### Data

- **`data/`**: Contains dataset files organized by patient splits
- **`checkpoints/`**: Stores trained model weights (per-fold or single model)

---

## Notes

- Models are automatically saved to the `checkpoints/` directory
- Cross-validation uses stratified k-fold splitting for consistent results
- Class indices are remapped to contiguous IDs (0=background) during multi-class training
- All training runs are logged to Weights & Biases for easy tracking and comparison

## Installation Guide

### Option 1: Using `uv` (Recommended)

`uv` is a fast, reliable Python package installer that improves upon pip and poetry. It's the recommended approach for this project.

#### 1.1 Install `uv`

For Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or install via your package manager:
```bash
# Homebrew (macOS)
brew install uv

# Cargo (if you have Rust)
cargo install uv
```

#### 1.2 Clone the Repository

```bash
git clone https://github.com/Dylanyves/UBPD.git
cd UBPD
```

#### 1.3 Install Dependencies

```bash
uv sync
```

This command will create a virtual environment and install all dependencies specified in `pyproject.toml`.

### Option 2: Using `pipenv`

If you prefer pipenv:

```bash
# Install pipenv if not already installed
pip install pipenv

# Clone and navigate to the project
git clone https://github.com/Dylanyves/UBPD.git
cd UBPD

# Install dependencies
pipenv install
pipenv shell  # Activate the virtual environment
```

---

## Usage Examples

### Training a Model

To train a U-Net model with cross-validation on all anatomical classes:

```bash
uv run main.py \
  --model unet \
  --epochs 100 \
  --batch_size 16 \
  --cv 5 \
  --include_classes 1 2 3 4 \
  --seed 42
```

To train a U-Net++ model focusing only on nerve and artery detection:

```bash
uv run main.py \
  --model unetpp \
  --epochs 150 \
  --batch_size 32 \
  --cv 5 \
  --include_classes 1 4 \
  --seed 42 \
  --use_wandb
```

Key parameters:
- `--model`: Model architecture (`unet` or `unetpp`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--cv`: Number of cross-validation folds
- `--include_classes`: Class IDs to include (1=artery, 2=vein, 3=muscle, 4=nerve)
- `--seed`: Random seed for reproducibility
- `--use_wandb`: Enable Weights & Biases tracking

### Evaluating a Model

To evaluate a trained model on the test set:

```bash
# Evaluate a 5-fold CV model
uv run evaluate.py \
  --model_id 616483 \
  --model_name unet \
  --cv \
  --include_classes 1 2 3 4
```

To evaluate a single model and visualize predictions:

```bash
# Evaluate with visualization
uv run evaluate.py \
  --model_id 616483 \
  --model_name unetpp \
  --include_classes 1 4 \
  --show_plot
```

Key parameters:
- `--model_id`: Model/experiment ID (matches checkpoint filename)
- `--model_name`: Model architecture used (`unet` or `unetpp`)
- `--cv`: Flag to evaluate multi-fold model
- `--include_classes`: Classes included in training
- `--show_plot`: Display prediction visualizations

---

## Project Structure

### Core Modules

```
src/
├── models/
│   ├── unet.py           # U-Net architecture implementation
│   └── unetpp.py         # U-Net++ architecture implementation
├── train.py              # Main Trainer class for training loop
├── evaluate.py           # Evaluator class for model evaluation
├── dataset.py            # UBPDataset class for data loading
├── train_utils.py        # Training utilities (loss, optimizer, metrics)
├── helper.py             # Helper functions (model building, transforms)
├── const.py              # Constants and paths
└── preprocessing.py      # Data preprocessing utilities
```

### Key Files

- **`main.py`**: Entry point for training. Orchestrates:
  - Cross-validation fold splitting
  - Dataset creation with class filtering and remapping
  - Training pipeline with Wandb logging
  - Test set evaluation and metric aggregation

- **`evaluate.py`**: Standalone evaluation CLI. Loads pre-trained checkpoints and evaluates on test set with metrics computation and visualization support

### Data

- **`data/`**: Contains dataset files organized by patient splits
- **`checkpoints/`**: Stores trained model weights (per-fold or single model)

---

## Notes

- Models are automatically saved to the `checkpoints/` directory
- Cross-validation uses stratified k-fold splitting for consistent results
- Class indices are remapped to contiguous IDs (0=background) during multi-class training
- All training runs are logged to Weights & Biases for easy tracking and comparison
