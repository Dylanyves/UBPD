**UBPD**

- **Purpose:**: Training, evaluation, and utilities for the UBPD segmentation experiments (PyTorch).

**Quick Start**
- **Python:**: Requires Python >= 3.10. Create and activate a virtual environment:
	`python -m venv .venv`  
	`source .venv/bin/activate`
- **Install:**: Install the project and dependencies from `pyproject.toml`:
	`python -m pip install -e .`

**Project Layout**
- **`main.py`**: Primary training CLI that runs cross-validation experiments.
- **`evaluate.py`**: CLI to evaluate saved checkpoints (single or CV-style).
- **`src/`**: Core code (dataset, models, training loop, evaluation helpers).
- **`checkpoints/`**: Stored model weights. Naming convention: `<model_id>_fold_<n>.pth` for CV runs or `<model_id>.pth` for single checkpoints.
- **`data/`**: Expected dataset root. Default paths are defined in `src/const.py` (`./data/dataset/images` and `./data/dataset/labels/json_train`).
- **`tests/`**, **`scripts.py`**, **`evaluate.py`**, **`evaluate.py`**: utilities and tests.

**Usage Examples**
- **Train (default):**
	`python main.py --model unet --cv --epochs 100 --batch_size 16 --image_size 256 --seed 42`
- **Evaluate (CV):**
	`python evaluate.py --model_id 616483 --model_name unet --cv --include_classes 1 2 3 4`
- **Evaluate (single checkpoint):**
	`python evaluate.py --model_id 616483 --model_name unet --include_classes 4`

**Notes / Conventions**
- **Classes:**: Landmark classes use IDs `1..4` (see `evaluate.py` and `main.py` for mappings). When a single class is selected the model uses a binary foreground channel; otherwise the pipeline remaps selected classes to contiguous IDs (background=0, classes=1..K).
- **Transforms & image size:**: Default image size is `256` (see `src/const.py`). Validation transforms are applied consistently between training and evaluation.
- **WandB:**: `main.py` supports logging to Weights & Biases; set `WANDB_API_KEY` and pass `--use_wandb` to enable.

**Where to look next**
- `src/train.py` — training loop and checkpoint saving.
- `src/evaluate.py` — evaluation utilities and plotting.
- `src/dataset.py` — dataset and transforms.

**Contact**
- For questions, open an issue or contact the repository owner.

