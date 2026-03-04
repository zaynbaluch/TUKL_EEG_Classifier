import numpy as np
from scipy.signal import lfilter, bessel
import pywt
from mspca import mspca

def bassel_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Original Bessel bandpass filter.
    """
    b, a = bessel(order, [lowcut, highcut], btype='band', fs=fs)
    y = lfilter(b, a, data)
    return y

def mspca_denoise(signal, wavelet='db4', level=5, model=None):
    """
    Multiscale PCA Denoising for single-channel signal using mspca library.
    
    Args:
        signal: 1D numpy array or [1, T] array
        wavelet: Wavelet family (default 'db4')
        level: Decomposition level (unused by library fit_transform)
        model: Pre-instantiated mspca.MultiscalePCA() object
        
    Returns:
        Denoised signal (same shape as input)
    """
    signal_np = np.array(signal)
    original_shape = signal_np.shape
    
    if signal_np.ndim == 1:
        signal_np = signal_np.reshape(-1, 1) # [T, 1]
    
    # Use provided model or create one if missing
    if model is None:
        model = mspca.MultiscalePCA()
    
    try:
        denoised_signal = model.fit_transform(signal_np, wavelet_func=wavelet)
        return denoised_signal.reshape(original_shape)
    except Exception as e:
        return signal

def vmd_decompose(signal, n_modes=5, alpha=2000, tau=0, tol=1e-7):
    """
    Variational Mode Decomposition of EEG signal.
    
    Args:
        signal: 1D numpy array
        n_modes: Number of IMFs
        alpha: Bandwidth constraint
        tau: Noise-tolerance (0 = clean)
        tol: Convergence tolerance
        
    Returns:
        IMFs as a 2D array [n_modes, signal_length]
    """
    try:
        from vmdpy import VMD
        # VMD(f, alpha, tau, K, DC, init, tol)
        # f: input signal (1D)
        # K: number of modes
        imfs, _, _ = VMD(signal, alpha, tau, n_modes, 0, 1, tol)
        return imfs  # shape: [n_modes, T]
    except ImportError:
        print("vmdpy not installed. Install with `pip install vmdpy`")
        return np.zeros((n_modes, len(signal)))
