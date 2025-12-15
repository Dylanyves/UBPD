# UBPD: Ultrasound Brachial Plexus Deep Learning Segmentation
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

To train a U-Net++ model focusing only on nerve and artery detection with Weights & Biases tracking:

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

#### Training Arguments

| Argument | Type | Default | Description | Options |
|----------|------|---------|-------------|---------|
| `--model` | str | `unet` | Model architecture to use | `unet`, `unetpp` |
| `--cv` | int | `5` | Number of cross-validation folds | Any positive integer |
| `--seed` | int | `42` | Random seed for reproducibility | Any integer |
| `--include_classes` | int (list) | `[1, 2, 3, 4]` | Class IDs to include in training | `1` (artery), `2` (vein), `3` (muscle), `4` (nerve) |
| `--epochs` | int | `100` | Number of training epochs | Any positive integer |
| `--batch_size` | int | `16` | Training batch size | Any positive integer |
| `--num-workers` | int | `2` | Number of data loading workers | Any non-negative integer |
| `--image_size` | int | `256` | Input image size (square) | Any positive integer |
| `--patience` | int | `15` | Early stopping patience (epochs without improvement) | Any positive integer |
| `--half_precision` | bool | `True` | Use FP16 mixed precision training | `True`, `False` |
| `--ignore_empty` | bool | `False` | Ignore empty (all-background) images during training | `True`, `False` |
| `--augment` | bool | `True` | Enable data augmentation | `True`, `False` |
| `--device` | str | `cuda` (if available) | Device to use for training | `cuda`, `cpu` |
| `--use_wandb` | bool | `False` | Enable Weights & Biases logging | `True`, `False` |

#### Weights & Biases (W&B) Setup

To log your training runs to Weights & Biases, use the `--use_wandb` flag. **Important**: You must first set up your W&B API key in a `.env` file:

1. Get your API key from [wandb.ai](https://wandb.ai)
2. Create a `.env` file in the project root:
   ```bash
   echo "WANDB_API_KEY=your_api_key_here" > .env
   ```
3. Run training with the flag:
   ```bash
   uv run main.py --use_wandb --model unet --cv 5
   ```

The training script will automatically log metrics, loss curves, and configurations to your W&B dashboard for easy tracking and comparison.

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

#### Evaluation Arguments

| Argument | Type | Default | Description | Options |
|----------|------|---------|-------------|---------|
| `--model_id` | str | **required** | Model identifier (matches checkpoint filename prefix) | Any string identifier |
| `--model_name` | str | `unet` | Model architecture used during training | `unet`, `unetpp` |
| `--cv` | flag | `False` | Evaluate all per-fold checkpoints | Set flag to enable |
| `--include_classes` | int (list) | `[1, 2, 3, 4]` | Class IDs to evaluate (must match training) | `1`, `2`, `3`, `4` |
| `--image_size` | int | `512` | Image size (must match training pipeline) | Any positive integer |
| `--seed` | int | `42` | Seed for reproducible train/test split | Any integer |
| `--device` | str | `cuda` (if available) | Device for evaluation | `cuda`, `cpu` |
| `--ignore_empty` | flag | `False` | Ignore empty (all-background) images for metrics | Set flag to enable |
| `--show_plot` | flag | `False` | Display example prediction visualizations | Set flag to enable |

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