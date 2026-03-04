import argparse
import yaml
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from datetime import datetime
import json

# Add src to python path for imports
sys.path.append(os.getcwd())

from src.data.dataset import EEGDataset
from src.models.conv_parallel import ConvParallelEEG1DModel
from src.training.trainer import train_epoch
from src.training.evaluator import evaluate

def merge_configs(base, override):
    """
    Recursively merge override dictionary into base dictionary.
    """
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def load_config(config_path):
    # Always load default first
    default_path = os.path.join("configs", "default.yaml")
    if not os.path.exists(default_path):
        print(f"Warning: {default_path} not found.")
        config = {}
    else:
        with open(default_path, 'r') as f:
            config = yaml.safe_load(f)
    
    if config_path != default_path:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                override = yaml.safe_load(f)
            if override:
                merge_configs(config, override)
        else:
             print(f"Error: Config file {config_path} not found.")
             sys.exit(1)
        
    return config


def log_confusion_matrix(writer, labels, predictions, class_names, epoch):
    """Log confusion matrix as a matplotlib figure to TensorBoard."""
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - Epoch {epoch}')
    writer.add_figure('Evaluation/confusion_matrix', fig, epoch)
    plt.close(fig)


def log_pr_curves(writer, labels, probs, class_names, epoch):
    """Log precision-recall curves per class to TensorBoard."""
    labels_np = np.array(labels)
    probs_np = np.array(probs)
    
    for i, class_name in enumerate(class_names):
        binary_labels = (labels_np == i).astype(int)
        class_probs = probs_np[:, i]
        
        writer.add_pr_curve(
            f'PR_Curve/{class_name}',
            binary_labels,
            class_probs,
            global_step=epoch
        )


def log_weight_histograms(writer, model, epoch):
    """Log model weight and gradient distributions to TensorBoard."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'Weights/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)


def main():
    parser = argparse.ArgumentParser(description="EEG Classification Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    
    # Optional CLI overrides (for sweeps or quick tests)
    parser.add_argument("--training-epochs", type=int, help="Override epochs")
    parser.add_argument("--training-learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--model-active-branches", type=str, help="Override active branches (e.g. '[0, 1]')")
    parser.add_argument("--dataset-limit", type=int, help="Limit dataset size for debugging")
    parser.add_argument("--training-device", type=str, help="Override device (cpu or cuda)")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.training_epochs:
        config['training']['epochs'] = args.training_epochs
    if args.training_learning_rate:
        config['training']['learning_rate'] = args.training_learning_rate
    if args.model_active_branches:
        config['model']['active_branches'] = eval(args.model_active_branches)
    if args.dataset_limit:
        config['data']['limit'] = args.dataset_limit
    if args.training_device:
        config['training']['device'] = args.training_device

    # Setup device
    device_str = config['training'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # TensorBoard Setup
    # ==========================================
    writer = None
    if config['tracker']['enabled']:
        # Create run directory with timestamp
        run_name = config['tracker'].get('run_name') or datetime.now().strftime('%Y%m%d_%H%M%S')
        project_name = config['tracker'].get('project_name', 'eeg-classification')
        log_dir = os.path.join(config['output']['log_dir'], 'tensorboard', f'{project_name}_{run_name}')
        os.makedirs(log_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"  Launch with: tensorboard --logdir {os.path.dirname(log_dir)}")
        
        # Log config as text
        config_text = json.dumps(config, indent=2, default=str)
        writer.add_text('Config/full_config', f'```json\n{config_text}\n```', 0)
    else:
        print("Experiment tracking disabled.")

    # Create datasets
    train_dataset = EEGDataset(config, mode='train')
    eval_dataset = EEGDataset(config, mode='eval')
    
    batch_size = config['training']['batch_size']
    eval_batch = config['training']['eval_batch']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(eval_dataset, batch_size=eval_batch, shuffle=False)
    
    # Model Setup
    print("Determining input channels from dataset sample...")
    try:
        sample_x1, sample_x2, sample_x3 = train_dataset[0][1:4]
        input_channels_list = [sample_x1.shape[0], sample_x2.shape[0], sample_x3.shape[0]]
        print(f"Input channels: {input_channels_list}")
    except Exception as e:
        print(f"Error loading sample: {e}")
        sys.exit(1)
        
    output_size = config['model']['output_size']
    num_classes = output_size
    class_names = [f'Class {i}' for i in range(num_classes)]
    
    model = ConvParallelEEG1DModel(input_channels_list, output_size)
    model.to(device)
    
    # Log model architecture
    if writer is not None:
        # Log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        writer.add_text('Model/architecture', 
                       f'Total Parameters: {total_params:,}\n'
                       f'Trainable Parameters: {trainable_params:,}\n'
                       f'Input Channels: {input_channels_list}\n'
                       f'Output Classes: {output_size}', 0)

    # Optimizer & Scheduler
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config['training']['scheduler_patience'], 
        factor=config['training']['scheduler_factor']
    )
    
    # Loss
    label_smoothing = config['training']['label_smoothing']
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    active_branches = config['model']['active_branches']
    aux_loss_weight = config['training']['aux_loss_weight']
    
    # Training Loop
    epochs = config['training']['epochs']
    best_eval_acc = 0.0
    best_eval_f1 = 0.0
    global_step = 0
    
    checkpoint_dir = config['output']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Active branches: {active_branches}")
    print(f"Aux loss weight: {aux_loss_weight}")
    
    for epoch in range(epochs):
        # ==========================================
        # Training
        # ==========================================
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, loss_fn, active_branches, device, 
            aux_loss_weight, writer=writer, epoch=epoch, global_step=global_step
        )
        
        # ==========================================
        # Validation
        # ==========================================
        eval_loss, eval_acc, eval_metrics = evaluate(model, val_loader, loss_fn, active_branches, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Eval Loss: {eval_loss:.4f} Acc: {eval_acc:.4f} | "
              f"F1: {eval_metrics['macro_f1']:.4f} | LR: {current_lr:.2e}")
        
        # ==========================================
        # TensorBoard Epoch-Level Logging
        # ==========================================
        if writer is not None:
            # --- Core Scalars ---
            writer.add_scalars('Loss/epoch', {
                'train': train_loss,
                'eval': eval_loss
            }, epoch)
            writer.add_scalars('Accuracy/epoch', {
                'train': train_acc,
                'eval': eval_acc
            }, epoch)
            writer.add_scalar('LearningRate/epoch', current_lr, epoch)
            
            # --- Per-Class Metrics (critical for academic papers) ---
            for c in range(num_classes):
                writer.add_scalar(f'PerClass/accuracy_class_{c}', eval_metrics[f'class_{c}_acc'], epoch)
                writer.add_scalar(f'PerClass/precision_class_{c}', eval_metrics[f'precision_{c}'], epoch)
                writer.add_scalar(f'PerClass/recall_class_{c}', eval_metrics[f'recall_{c}'], epoch)
                writer.add_scalar(f'PerClass/f1_class_{c}', eval_metrics[f'f1_{c}'], epoch)
            
            # --- Macro Averages ---
            writer.add_scalar('Macro/precision', eval_metrics['macro_precision'], epoch)
            writer.add_scalar('Macro/recall', eval_metrics['macro_recall'], epoch)
            writer.add_scalar('Macro/f1', eval_metrics['macro_f1'], epoch)
            
            # --- Weight & Gradient Histograms ---
            log_weight_histograms(writer, model, epoch)
            
            # --- Confusion Matrix (as figure) ---
            # Collect predictions for confusion matrix and PR curves
            all_preds = []
            all_labels = []
            all_probs = []
            
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    _, x1, x2, x3, y = batch
                    x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
                    output, _ = model([x1, x2, x3], active_branch_indices=active_branches)
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            log_confusion_matrix(writer, all_labels, all_preds, class_names, epoch)
            log_pr_curves(writer, all_labels, all_probs, class_names, epoch)
            
            writer.flush()
            
        # Scheduler Step
        scheduler.step(eval_loss)
        
        # Save Checkpoint
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            ckpt_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'eval_acc': eval_acc,
                'eval_loss': eval_loss,
                'eval_metrics': eval_metrics,
                'config': config,
            }, ckpt_path)
            print(f"  ✓ Saved best model (Acc: {best_eval_acc:.4f})")
        
        if eval_metrics['macro_f1'] > best_eval_f1:
            best_eval_f1 = eval_metrics['macro_f1']
            ckpt_path = os.path.join(checkpoint_dir, "best_f1.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'eval_acc': eval_acc,
                'eval_loss': eval_loss,
                'eval_metrics': eval_metrics,
                'config': config,
            }, ckpt_path)
            print(f"  ✓ Saved best F1 model (Macro F1: {best_eval_f1:.4f})")

    # ==========================================
    # Final HParams Logging (for experiment comparison)
    # ==========================================
    if writer is not None:
        hparam_dict = {
            'lr': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'epochs': config['training']['epochs'],
            'weight_decay': config['training']['weight_decay'],
            'label_smoothing': config['training']['label_smoothing'],
            'aux_loss_weight': config['training']['aux_loss_weight'],
            'active_branches': str(config['model']['active_branches']),
            'preprocessing': config['data'].get('preprocessing', 'bessel'),
            'feature_branch_2': config['data'].get('feature_branch_2', 'wavelet'),
            'hidden_size': config['model'].get('hidden_size', 20),
        }
        metric_dict = {
            'hparam/best_eval_acc': best_eval_acc,
            'hparam/best_eval_f1': best_eval_f1,
        }
        writer.add_hparams(hparam_dict, metric_dict)
        writer.close()
        print(f"\nTensorBoard logs saved. View with: tensorboard --logdir {os.path.dirname(log_dir)}")

    print(f"\nTraining complete. Best Eval Acc: {best_eval_acc:.4f}, Best Macro F1: {best_eval_f1:.4f}")

if __name__ == "__main__":
    main()
