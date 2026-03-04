import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pywt
from scipy.signal import lfilter, bessel, stft

class EEGDataset(Dataset):
    def __init__(self, files_list, number_of_eeg_channels):
        self.csv_file_path = files_list
        self.file_list = []
        self.number_of_eeg_channels = number_of_eeg_channels
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.csv_file_path)
        df['file_name'] = df['file_name']#.str.replace("SSD 4TB", "SSD 4TB1", regex=False)

        self.file_list = df['file_name'].tolist()
        self.ab_label = df['abnormality_type_3_class'].tolist()

    def __len__(self):
        return len(self.file_list)

    def get_all_labels(self):
        return self.ab_label  # already parsed in __init__

    def __getitem__(self, idx):
        npz_data = np.load(self.file_list[idx])
        eeg_data = npz_data['data'][0:self.number_of_eeg_channels]

        # fft_result = np.fft.fft(bassel_bandpass_filter(eeg_data, 0.01, 15, 200, order=4))
        # power_spectrum = np.abs(fft_result)**2        
        # power_spectrum = (power_spectrum - np.min(power_spectrum)) / (np.max(power_spectrum) - np.min(power_spectrum))

        filtered = bassel_bandpass_filter(eeg_data, 0.01, 15, 200, order=4)
        spec_list = []
        for channel_data in filtered:
            f, t, Zxx = stft(channel_data, fs=200, nperseg=64, noverlap=32)
            power = np.abs(Zxx)**2  # shape: [freq_bins, time_steps]
            spec_list.append(power)

        # Stack over channels → [channels, freq, time]
        spec_array = np.stack(spec_list, axis=0)

        # Optional: Average over channels or keep per-channel
        power_spectrum = np.mean(spec_array, axis=0)  # shape: [freq, time]
        
        # Perform Continuous Wavelet Transform (CWT)
        coefficients_cmor, frequencies_cmor = pywt.cwt(eeg_data, np.arange(1, 31), 'cmor1.5-1.0', sampling_period=1/200)
        # coefficients_cmor, frequencies_cmor = pywt.cwt(eeg_data, np.arange(1, 31), 'gmw', sampling_period=1/200)

        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        eeg_data = eeg_data.unsqueeze(0)

        power_spectrum = torch.tensor(power_spectrum, dtype=torch.float32).unsqueeze(0)  # [1, F, T]

        # coefficients_cmor: [C, scales, T]
        coefficients_cmor = torch.tensor(coefficients_cmor, dtype=torch.float32)

        wavelet_transform = torch.tensor(coefficients_cmor, dtype=torch.float32)
        wavelet_transform = wavelet_transform.squeeze(1)
        wavelet_transform = wavelet_transform.unsqueeze(0)

        # print(eeg_data.shape)
        # print(power_spectrum.shape)
        # print(wavelet_transform.shape)

        label = self.ab_label[idx]
        return self.file_list[idx], eeg_data, power_spectrum, wavelet_transform, torch.tensor(label, dtype=torch.float32)

def bassel_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = bessel(order, [lowcut, highcut], btype='band', fs=fs)
        y = lfilter(b, a, data)
        return y