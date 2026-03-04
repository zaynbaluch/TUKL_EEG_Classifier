# EEG Abnormality Classification

A multi-branch 1D Convolutional Neural Network (CNN) for classifying EEG signals into 3 abnormality types. The model processes Raw EEG, STFT power spectra, and VMD/Wavelet features in parallel branches.

## Key Features

- **Multi-Branch Model**: Combines Raw EEG + STFT + VMD (or Wavelet) features via parallel Conv1D branches.
- **Preprocessing**: Custom GPU-accelerated MSPCA denoising (PyTorch SVD) or Bessel bandpass filtering.
- **Experiment Tracking**: Integrated with **TensorBoard** — logs scalars, per-class metrics, confusion matrices, PR curves, weight histograms, and hyperparameters.
- **Visualization**: t-SNE plots per branch and combined, with configurable subsample size.

## Directory Structure

```
EEG/
├── configs/                    # YAML experiment configs
│   ├── default.yaml            # Base config (all options)
│   ├── experiment_branch_0.yaml
│   ├── experiment_branch_01.yaml
│   └── experiment_branch_012.yaml
├── src/                        # Source code
│   ├── data/                   # Dataset, preprocessing (MSPCA, Bessel, VMD)
│   ├── models/                 # Model architecture (ConvParallelEEG1DModel)
│   ├── training/               # Training and evaluation loops
│   └── visualization/          # t-SNE plotting
├── scripts/                    # CLI scripts
│   ├── train.py                # Training with TensorBoard logging
│   ├── test.py                 # Evaluation with classification report
│   └── visualize.py            # t-SNE visualization
├── outputs/                    # Run outputs (checkpoints, logs, plots)
└── dataset/                    # CSV label files + data paths
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**: Place `train.csv`, `eval.csv`, `test.csv` in `dataset/`. Each CSV must have `file_name` (path to `.npz`) and `abnormality_type_3_class` (label) columns.

## Usage

### Training

Train with the default config:
```bash
python scripts/train.py --config configs/default.yaml
```

Run a specific branch experiment:
```bash
python scripts/train.py --config configs/experiment_branch_012.yaml
```

Override parameters via CLI:
```bash
python scripts/train.py --config configs/default.yaml --training-epochs 100 --model-active-branches "[0, 1]"
```

### Viewing TensorBoard Logs

```bash
tensorboard --logdir outputs/logs/tensorboard
```

### Evaluation

```bash
python scripts/test.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

### t-SNE Visualization

```bash
python scripts/visualize.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

## Configuration Reference

All options are set in `configs/default.yaml`. Experiment configs (e.g., `experiment_branch_01.yaml`) override only the fields they specify.

### `data`

| Key | Default | Options | Description |
|---|---|---|---|
| `train_csv` | `./dataset/train.csv` | Any path | Path to training CSV |
| `eval_csv` | `./dataset/eval.csv` | Any path | Path to validation CSV |
| `test_csv` | `./dataset/test.csv` | Any path | Path to test CSV |
| `num_eeg_channels` | `1` | Integer | Number of EEG channels to load from `.npz` |
| `preprocessing` | `mspca` | `mspca`, `bessel` | Denoising method. MSPCA uses GPU-accelerated SVD; Bessel uses a bandpass filter (0.01–15 Hz) |
| `feature_branch_2` | `vmd` | `vmd`, `wavelet` | Feature type for Branch 2. VMD = Variational Mode Decomposition; Wavelet = CWT (`cmor1.5-1.0`) |
| `vmd_modes` | `5` | Integer | Number of IMFs when using VMD |

### `model`

| Key | Default | Options | Description |
|---|---|---|---|
| `output_size` | `3` | Integer | Number of classification classes |
| `hidden_size` | `20` | Integer | Hidden layer size |
| `active_branches` | `[0]` | `[0]`, `[0,1]`, `[0,1,2]` | Which branches to activate. 0 = Raw EEG, 1 = STFT, 2 = VMD/Wavelet |

### `training`

| Key | Default | Options | Description |
|---|---|---|---|
| `epochs` | `50` | Integer | Number of training epochs |
| `batch_size` | `4` | Integer | Training batch size |
| `eval_batch` | `4` | Integer | Evaluation batch size |
| `learning_rate` | `0.0001` | Float | Initial learning rate (AdamW) |
| `weight_decay` | `0.0001` | Float | AdamW weight decay |
| `label_smoothing` | `0.1` | 0.0–1.0 | Label smoothing for CrossEntropyLoss |
| `aux_loss_weight` | `0.3` | Float | Weight for auxiliary branch losses |
| `scheduler_patience` | `3` | Integer | ReduceLROnPlateau patience |
| `scheduler_factor` | `0.5` | Float | LR reduction factor |
| `device` | `cuda` | `cuda`, `cpu` | Training device |

### `tracker`

| Key | Default | Options | Description |
|---|---|---|---|
| `enabled` | `true` | `true`, `false` | Enable/disable TensorBoard logging |
| `backend` | `tensorboard` | `tensorboard` | Tracking backend |
| `project_name` | `eeg-abnormality-classification` | String | Project name (used in log directory) |
| `run_name` | `null` | String or `null` | Run name. Auto-generated timestamp if `null` |

### `output`

| Key | Default | Description |
|---|---|---|
| `checkpoint_dir` | `./outputs/checkpoints` | Where model checkpoints are saved |
| `prediction_dir` | `./outputs/predictions` | Where test predictions CSV is saved |
| `log_dir` | `./outputs/logs` | TensorBoard log directory |
| `plot_dir` | `./outputs/plots` | Where t-SNE and confusion matrix plots are saved |

### `visualization`

| Key | Default | Description |
|---|---|---|
| `tsne_subsample_size` | `2000` | Number of samples for t-SNE (for speed) |
| `tsne_perplexity` | `30` | t-SNE perplexity parameter |
| `tsne_random_state` | `42` | Random seed for reproducibility |

## TensorBoard Logged Metrics

| Category | Metrics |
|---|---|
| **Loss** | Train/eval loss per epoch, batch-level loss |
| **Accuracy** | Train/eval accuracy per epoch, per-class accuracy |
| **Per-Class** | Precision, recall, F1 for each class |
| **Macro** | Macro precision, recall, F1 |
| **Figures** | Confusion matrix per epoch |
| **PR Curves** | Precision-recall curve per class |
| **Histograms** | Weight and gradient distributions per layer |
| **HParams** | Full hyperparameter comparison across runs |
