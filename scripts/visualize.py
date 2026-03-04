import argparse
import yaml
import os
import sys
import torch
from torch.utils.data import DataLoader
import wandb

# Add src to python path for imports
sys.path.append(os.getcwd())

from src.data.dataset import EEGDataset
from src.models.conv_parallel import ConvParallelEEG1DModel
from src.visualization.tsne import plot_tsne

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="EEG Classification Visibility")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    device_str = config['training'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset (Use Test set for final visualization usually, or Eval)
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
    tsne_cfg = config.get('visualization', {})
    subsample_size = tsne_cfg.get('tsne_subsample_size', 2000)
    
    # Collect features
    all_features = {} # {'branch_0': [], ...}
    all_labels = []

    print("Extracting features...")
    with torch.no_grad():
        for batch in test_loader:
            _, x1, x2, x3, y = batch
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
            
            feats_dict = model.extract_features([x1, x2, x3], active_branch_indices=active_branches)
            
            for k, v in feats_dict.items():
                if k not in all_features:
                    all_features[k] = []
                all_features[k].append(v.cpu().numpy())
            
            all_labels.extend(y.numpy())

    # Concatenate features
    for k in all_features:
        all_features[k] = np.vstack(all_features[k])
    
    all_labels = np.array(all_labels)
    
    plot_dir = config['output']['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate Plots
    for k, feats in all_features.items():
        if "branch" in k:
             # Check if branch was active (feats should be non-zero if active)
             # If branch index > len(active_branches), it's inactive? The dict keys are "branch_idx".
             # Model inserts zeros for inactive branches.
             # We can skip plotting if branch is known inactive or if features are all zero.
             if np.all(feats == 0):
                 print(f"Skipping plot for {k} (inactive/zero features)")
                 continue
        
        save_path = os.path.join(plot_dir, f"tsne_{k}.png")
        plot_tsne(feats, all_labels, f"t-SNE - {k}", save_path, 
                  subsample_size=subsample_size, 
                  perplexity=tsne_cfg.get('tsne_perplexity', 30),
                  random_state=tsne_cfg.get('tsne_random_state', 42))

if __name__ == "__main__":
    main()
