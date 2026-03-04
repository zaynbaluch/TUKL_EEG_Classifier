# EEG Abnormality Classification

This project implements a multi-branch 1D Convolutional Neural Network (CNN) to classify EEG signals into 3 abnormality types. The model processes raw EEG data, STFT power spectra, and VMD (Variational Mode Decomposition) features in parallel branches.

## Key Features

- **Multi-Input Model**: Combines Raw EEG + STFT + VMD features.
- **Preprocessing**: MSPCA (Multi-Scale Principal Component Analysis) for denoising.
- **Experiment Tracking**: Integrated with **Weights & Biases (wandb)**.
- **Visualization**: t-SNE plots for class separation analysis.

## Directory Structure

```
EEG/
├── configs/                    # YAML experiment configs
├── src/                        # Source code
│   ├── data/                   # Dataset and preprocessing
│   ├── models/                 # Model architecture
│   ├── training/               # Training and evaluation loops
│   └── visualization/          # t-SNE plotting
├── scripts/                    # CLI scripts (train.py, visualize.py)
├── outputs/                    # Run outputs (models, logs, plots)
└── dataset/                    # Data files
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Login to wandb** (optional, for experiment tracking):
   ```bash
   wandb login
   ```

## Usage

### Training
Train the model using the default configuration:
```bash
python scripts/train.py --config configs/default.yaml
```

Override parameters via CLI:
```bash
python scripts/train.py --config configs/default.yaml --training.epochs 100 --model.active_branches "[0, 1]"
```

### Evaluation
Evaluate a trained model on the test set:
```bash
python scripts/test.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

### Visualization
Generate t-SNE plots for a trained model:
```bash
python scripts/visualize.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```
