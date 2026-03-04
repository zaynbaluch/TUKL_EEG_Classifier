import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.signal import stft
import pywt
from src.data.preprocessing import bassel_bandpass_filter, mspca_denoise, vmd_decompose

class EEGDataset(Dataset):
    def __init__(self, config, mode='train'):
        """
        Args:
            config: Configuration dictionary (or object)
            mode: 'train', 'eval', or 'test' to select correct CSV
        """
        self.config = config
        self.mode = mode
        
        # Select CSV file based on mode
        if mode == 'train':
            self.csv_path = config['data']['train_csv']
        elif mode == 'eval':
            self.csv_path = config['data']['eval_csv']
        elif mode == 'test':
            self.csv_path = config['data']['test_csv']
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        self.num_channels = config['data']['num_eeg_channels']
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        # Ensure file paths are strings
        self.file_list = df['file_name'].astype(str).tolist()
        self.labels = df['abnormality_type_3_class'].tolist()
        
        # Preprocessing settings
        self.preprocess_method = config['data'].get('preprocessing', 'bessel')
        self.feature_branch_2 = config['data'].get('feature_branch_2', 'wavelet')
        self.vmd_modes = config['data'].get('vmd_modes', 5)

        # Device for GPU-accelerated MSPCA
        device_str = config['training'].get('device', 'cuda')
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load .npz file
        file_path = self.file_list[idx]
        try:
            npz_data = np.load(file_path)
            eeg_data = npz_data['data'][0:self.num_channels] # [C, T]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros in case of error (or handle differently)
            # Assuming T=400 based on previous files, but better to fail or skip
            # For now, let's raise to fail fast or return None (dataloader might crash)
            raise e

        # 1. Preprocessing (Denoising)
        if self.preprocess_method == 'mspca':
            # mspca_denoise expects 1D or 2D. 
            # If eeg_data is [1, T], pass [T] or [1, T]
            # Our function handles it.
            # We iterate over channels if C > 1, but here C=1 typically.
            clean_data_list = []
            for ch in range(eeg_data.shape[0]):
                clean_ch = mspca_denoise(eeg_data[ch], device=self.device)
                clean_data_list.append(clean_ch.flatten())
            eeg_data = np.stack(clean_data_list) # [C, T]
            
        elif self.preprocess_method == 'bessel':
            eeg_data = bassel_bandpass_filter(eeg_data, 0.01, 15, 200, order=4)
        
        # --- Feature Branch 0: Raw EEG ---
        # Returns [C, T]
        x1 = torch.tensor(eeg_data, dtype=torch.float32)

        # --- Feature Branch 1: STFT Power ---
        # Same logic as original but using cleaned data
        spec_list = []
        for ch in range(eeg_data.shape[0]):
            f, t, Zxx = stft(eeg_data[ch], fs=200, nperseg=64, noverlap=32)
            power = np.abs(Zxx)**2
            spec_list.append(power) 
        
        spec_array = np.stack(spec_list, axis=0) # [C, F, T_stft]
        # Average over channels if C > 1 to keep dimensions simplified? 
        # Original code did np.mean(spec_array, axis=0)
        # But if we have C=1, np.mean removes axis 0.
        # Let's keep distinct channels if possible, or average.
        # Original code: power_spectrum = np.mean(spec_array, axis=0) -> [F, T_stft]
        # Then unsqueeze(0) -> [1, F, T_stft]
        # Let's replicate original behavior for consistency
        x2 = np.mean(spec_array, axis=0) 
        x2 = torch.tensor(x2, dtype=torch.float32) # [F=33, T=14]
        
        # --- Feature Branch 2: Wavelet or VMD ---
        if self.feature_branch_2 == 'vmd':
            # VMD decomposition
            # Valid only for C=1 for now as per VMD constraints usually
            # If C>1, we might need average or per-channel VMD
            # Using 1st channel for VMD if multiple
            imfs = vmd_decompose(eeg_data[0], n_modes=self.vmd_modes)
            x3 = torch.tensor(imfs, dtype=torch.float32) # [n_modes, T]
            
        else: # 'wavelet'
            # CWT
            # Original: pywt.cwt(eeg_data, ...)
            # Takes eeg_data [C, T] or [T]? 
            # Original code passed `eeg_data` (numpy) which was [1, T].
            # pywt.cwt returns [scales, 1, T] if input [1, T].
            # Then squeeze(1) -> [scales, T]. 
            # Then unsqueeze(0) -> [1, scales, T].
            
            # Let's compute CWT on the first channel to be safe/simple
            signal = eeg_data[0]
            coefficients, frequencies = pywt.cwt(signal, np.arange(1, 31), 'cmor1.5-1.0', sampling_period=1/200)
            x3 = torch.tensor(coefficients, dtype=torch.float32) # [scales=30, T]
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return self.file_list[idx], x1, x2, x3, label
