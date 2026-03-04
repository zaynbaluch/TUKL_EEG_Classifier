import argparse
import yaml
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())

from src.data.dataset import EEGDataset
from src.models.conv_parallel import ConvParallelEEG1DModel

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="EEG Classification Testing")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_csv", type=str, default="test_results.csv", help="Path to save results CSV")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    device_str = config['training'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Test Dataset
    test_dataset = EEGDataset(config, mode='test')
    batch_size = config['training']['eval_batch']
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    sample_x1, sample_x2, sample_x3 = test_dataset[0][1:4]
    input_channels_list = [sample_x1.shape[0], sample_x2.shape[0], sample_x3.shape[0]]
    output_size = config['model']['output_size']
    
    model = ConvParallelEEG1DModel(input_channels_list, output_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    active_branches = config['model']['active_branches']
    
    all_preds = []
    all_labels = []
    all_probs = []
    file_names = []
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
             fname, x1, x2, x3, y = batch
             x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
             
             output, _ = model([x1, x2, x3], active_branch_indices=active_branches)
             probs = torch.softmax(output, dim=1)
             preds = torch.argmax(probs, dim=1)
             
             all_preds.extend(preds.cpu().numpy())
             all_labels.extend(y.numpy())
             all_probs.extend(probs.cpu().numpy())
             file_names.extend(fname)
             
    # Metrics
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1', 'Class 2'], digits=4)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plot_path = os.path.join(config['output']['plot_dir'], "confusion_matrix.png")
    os.makedirs(config['output']['plot_dir'], exist_ok=True)
    plt.savefig(plot_path)
    print(f"Confusion Matrix saved to {plot_path}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        "file_name": file_names,
        "label": all_labels,
        "prediction": all_preds,
        "prob_0": [p[0] for p in all_probs],
        "prob_1": [p[1] for p in all_probs],
        "prob_2": [p[2] for p in all_probs]
    })
    
    pred_dir = config['output']['prediction_dir']
    os.makedirs(pred_dir, exist_ok=True)
    save_path = os.path.join(pred_dir, args.output_csv)
    results_df.to_csv(save_path, index=False)
    print(f"Detailed results saved to {save_path}")

if __name__ == "__main__":
    main()
