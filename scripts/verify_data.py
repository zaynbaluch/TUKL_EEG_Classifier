import argparse
import yaml
import os
import sys
import time
import torch
import numpy as np
import psutil

# Add src to python path for imports
sys.path.append(os.getcwd())

from src.data.dataset import EEGDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Verify Data Loading")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--index", type=int, default=0, help="Index of sample to verify")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    print(f"Loading configuration from {args.config}")
    
    # Force CPU for verification
    config['training']['device'] = 'cpu'
    
    print("Initializing Dataset...")
    dataset = EEGDataset(config, mode='train')
    print(f"Dataset length: {len(dataset)}")
    
    idx = args.index
    print(f"\nProcessing sample at index {idx}...")
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"Memory before: {mem_before:.2f} MB")
    
    start_time = time.time()
    try:
        sample = dataset[idx]
        # Unpack
        file_name, x1, x2, x3, label = sample
        
        duration = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory after: {mem_after:.2f} MB")
        print(f"Preprocessing took: {duration:.4f} seconds")
        
        print("\nShapes:")
        print(f"x1 (Raw): {x1.shape}")
        print(f"x2 (STFT): {x2.shape}")
        print(f"x3 (VMD/Wavelet): {x3.shape}")
        print(f"Label: {label}")
        print(f"File: {file_name}")
        
    except Exception as e:
        print(f"\nERROR processing sample: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
