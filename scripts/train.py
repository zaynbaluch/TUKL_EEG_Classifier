import argparse
import yaml
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import numpy as np

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
        # Fallback if running from scripts dir? No, os.getcwd() is root.
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

def main():
    parser = argparse.ArgumentParser(description="EEG Classification Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    
    # Optional CLI overrides (for sweeps or quick tests)
    # Use hyphens in arguments, argparse converts to underscores in namespace
    parser.add_argument("--training-epochs", type=int, help="Override epochs")
    parser.add_argument("--training-learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--model-active-branches", type=str, help="Override active branches (e.g. '[0, 1]')")
    parser.add_argument("--dataset-limit", type=int, help="Limit dataset size for debugging") # Optional useful feature
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

    # Initialize WandB
    if config['tracker']['enabled']:
        # Check if login is needed? Usually assumes user is logged in.
        wandb.init(
            project=config['tracker']['project_name'],
            name=config['tracker']['run_name'],
            config=config,
            dir=config['output']['log_dir']
        )
    else:
        print("WandB tracking disabled.")

    # Create datasets
    train_dataset = EEGDataset(config, mode='train')
    eval_dataset = EEGDataset(config, mode='eval')
    # test_dataset = EEGDataset(config, mode='test') # Not used in training loop, save memory
    
    batch_size = config['training']['batch_size']
    eval_batch = config['training']['eval_batch']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(eval_dataset, batch_size=eval_batch, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=eval_batch, shuffle=False)
    
    # Model Setup
    # Determine input channels from dataset
    print("Determining input channels from dataset sample...")
    try:
        sample_x1, sample_x2, sample_x3 = train_dataset[0][1:4]
        input_channels_list = [sample_x1.shape[0], sample_x2.shape[0], sample_x3.shape[0]]
        print(f"Input channels: {input_channels_list}")
    except Exception as e:
        print(f"Error loading sample: {e}")
        sys.exit(1)
        
    output_size = config['model']['output_size']
    
    model = ConvParallelEEG1DModel(input_channels_list, output_size)
    model.to(device)
    
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
    
    checkpoint_dir = config['output']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, active_branches, device, 
            aux_loss_weight, enable_tracker=config['tracker']['enabled']
        )
        
        # Validation
        eval_loss, eval_acc, eval_metrics = evaluate(model, val_loader, loss_fn, active_branches, device)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Eval Loss: {eval_loss:.4f} Acc: {eval_acc:.4f}")
        
        # Logging
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "lr": optimizer.param_groups[0]['lr']
        }
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        
        if config['tracker']['enabled']:
            wandb.log(metrics)
            
        # Scheduler Step
        scheduler.step(eval_loss)
        
        # Save Checkpoint
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            ckpt_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model with Acc: {best_eval_acc:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
