"""
Utility: Quality metrics — PSNR, SSIM, compression ratio
"""

import numpy as np
import os

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def psnr(original: np.ndarray, reconstructed: np.ndarray,
         max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better; > 30 dB is generally considered good.
    """
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    # Reshape if needed (e.g. single-channel)
    if orig.shape != recon.shape:
        h = min(orig.shape[0], recon.shape[0])
        w = min(orig.shape[1], recon.shape[1])
        orig  = orig[:h, :w]
        recon = recon[:h, :w]
    mse = np.mean((orig - recon) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10((max_val ** 2) / mse)


def ssim_score(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Structural Similarity Index (0 – 1).
    Requires scikit-image. Returns NaN if unavailable.
    """
    if not _HAS_SKIMAGE:
        return float('nan')
    orig  = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    if orig.shape != recon.shape:
        h = min(orig.shape[0], recon.shape[0])
        w = min(orig.shape[1], recon.shape[1])
        orig  = orig[:h, :w]
        recon = recon[:h, :w]
    channel_axis = -1 if orig.ndim == 3 else None
    return float(_ssim(orig, recon, data_range=255.0,
                       channel_axis=channel_axis))


def compression_ratio(original_bytes: int, compressed_bytes: int) -> dict:
    """
    Compute compression statistics.

    Returns
    -------
    dict with keys:
      ratio         — original / compressed  (higher = better)
      bits_per_pixel — compressed bits / total pixels (not applicable for multi-frame)
      space_saved   — % space saved
    """
    if compressed_bytes == 0:
        return {'ratio': float('inf'), 'bits_per_pixel': 0, 'space_saved': 100.0}
    ratio = original_bytes / compressed_bytes
    saved = (1 - compressed_bytes / original_bytes) * 100
    return {
        'ratio':       round(ratio, 3),
        'space_saved': round(saved, 2),
    }
