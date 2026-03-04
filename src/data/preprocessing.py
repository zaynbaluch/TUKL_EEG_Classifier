import numpy as np
from scipy.signal import lfilter, bessel
import pywt
import torch

def bassel_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Original Bessel bandpass filter.
    """
    b, a = bessel(order, [lowcut, highcut], btype='band', fs=fs)
    y = lfilter(b, a, data)
    return y


def mspca_denoise(signal, wavelet='db4', level=5, n_components=None, device=None):
    """
    Custom Multi-Scale Principal Component Analysis (MSPCA) denoising.
    
    GPU-accelerated implementation using PyTorch for the PCA (SVD) step,
    while using PyWavelets for wavelet decomposition/reconstruction.
    
    Algorithm:
        1. Wavelet decomposition (pywt.wavedec) to obtain multi-scale coefficients
        2. For each scale level, apply PCA via truncated SVD to denoise:
           - Center the coefficient vector
           - Compute SVD (on GPU if available)
           - Retain only top principal components (variance-based thresholding)
           - Reconstruct denoised coefficients
        3. Inverse wavelet transform (pywt.waverec) to reconstruct the signal
    
    Args:
        signal: 1D numpy array [T] or 2D [1, T]
        wavelet: Wavelet family string (default 'db4')
        level: Decomposition level (default 5)
        n_components: Number of PCA components to retain per scale.
                      If None, uses automatic variance-based thresholding (retains 95% variance).
        device: torch device string ('cuda', 'cpu') or torch.device. 
                If None, auto-detects GPU.
    
    Returns:
        Denoised signal as numpy array (same shape as input)
    """
    signal_np = np.array(signal, dtype=np.float64)
    original_shape = signal_np.shape
    
    # Flatten to 1D for processing
    if signal_np.ndim == 2:
        signal_np = signal_np.flatten()
    
    # Auto-detect device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # Step 1: Wavelet decomposition
    # Returns [cA_n, cD_n, cD_(n-1), ..., cD_1]
    # where cA_n = approximation coefficients at level n
    #       cD_i = detail coefficients at level i
    coeffs = pywt.wavedec(signal_np, wavelet, level=level)
    
    # Step 2: PCA denoising at each scale
    denoised_coeffs = []
    for i, coeff in enumerate(coeffs):
        denoised = _pca_denoise_coefficients(coeff, n_components=n_components, device=device)
        denoised_coeffs.append(denoised)
    
    # Step 3: Inverse wavelet transform
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    
    # Trim to original length (waverec may pad by 1 sample)
    reconstructed = reconstructed[:len(signal_np)]
    
    return reconstructed.reshape(original_shape)


def _pca_denoise_coefficients(coefficients, n_components=None, variance_threshold=0.95, device=None):
    """
    Apply PCA-based denoising to a 1D array of wavelet coefficients.
    
    For a single-channel signal, we construct a Hankel-like (trajectory) matrix
    from the coefficient vector to create a multi-dimensional representation
    suitable for PCA. This is the standard approach for single-channel MSPCA
    (similar to SSA — Singular Spectrum Analysis).
    
    Args:
        coefficients: 1D numpy array of wavelet coefficients at one scale
        n_components: Number of components to retain. If None, auto-select via variance threshold.
        variance_threshold: Fraction of variance to retain (default 0.95, i.e., 95%)
        device: torch.device for SVD computation
        
    Returns:
        Denoised 1D numpy array of same length
    """
    coeff_len = len(coefficients)
    
    # For very short coefficient vectors, skip PCA (not enough data)
    if coeff_len < 4:
        return coefficients
    
    # Window size for trajectory matrix (embedding dimension)
    # Use a window that is roughly half the length, but capped for efficiency
    window_size = min(coeff_len // 2, 32)
    if window_size < 2:
        return coefficients
    
    # Build trajectory (Hankel) matrix: shape [K, window_size]
    # where K = coeff_len - window_size + 1
    K = coeff_len - window_size + 1
    trajectory = np.zeros((K, window_size), dtype=np.float64)
    for j in range(window_size):
        trajectory[:, j] = coefficients[j:j + K]
    
    # Center the matrix (subtract column means)
    col_means = trajectory.mean(axis=0)
    trajectory_centered = trajectory - col_means
    
    # Move to GPU for SVD
    traj_tensor = torch.tensor(trajectory_centered, dtype=torch.float64, device=device)
    
    try:
        U, S, Vh = torch.linalg.svd(traj_tensor, full_matrices=False)
    except Exception:
        # Fallback to CPU if GPU SVD fails
        traj_tensor = traj_tensor.cpu()
        U, S, Vh = torch.linalg.svd(traj_tensor, full_matrices=False)
    
    # Determine number of components to retain
    if n_components is None:
        # Variance-based thresholding
        singular_values = S.cpu().numpy()
        total_variance = np.sum(singular_values ** 2)
        cumulative_variance = np.cumsum(singular_values ** 2) / total_variance
        n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
        n_components = max(1, min(n_components, len(singular_values)))
    else:
        n_components = min(n_components, S.shape[0])
    
    # Reconstruct with truncated SVD
    U_trunc = U[:, :n_components]
    S_trunc = torch.diag(S[:n_components])
    Vh_trunc = Vh[:n_components, :]
    
    reconstructed_tensor = U_trunc @ S_trunc @ Vh_trunc
    reconstructed_matrix = reconstructed_tensor.cpu().numpy() + col_means
    
    # Average the anti-diagonals to recover the 1D signal (Hankelization)
    denoised = _hankel_averaging(reconstructed_matrix, coeff_len)
    
    return denoised


def _hankel_averaging(matrix, original_length):
    """
    Recover a 1D signal from a trajectory matrix by averaging anti-diagonals.
    This is the standard diagonal averaging (Hankelization) step used in SSA/MSPCA.
    
    Args:
        matrix: 2D numpy array [K, L] (trajectory matrix)
        original_length: Length of the original 1D signal (= K + L - 1)
    
    Returns:
        1D numpy array of length original_length
    """
    K, L = matrix.shape
    N = original_length
    result = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)
    
    for i in range(K):
        for j in range(L):
            result[i + j] += matrix[i, j]
            counts[i + j] += 1.0
    
    result /= counts
    return result


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
