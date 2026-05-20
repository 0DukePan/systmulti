"""
Stage 1 — Pre-processing
BGR ↔ YCbCr conversion + 4:2:0 chroma subsampling
"""

import numpy as np
import cv2


# ─── Encoder ──────────────────────────────────────────────────────────────────

def bgr_to_ycbcr(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a BGR uint8 image to three float32 channels: Y, Cb, Cr.
    Uses the ITU-R BT.601 matrix (same as JPEG / MPEG).
    """
    # OpenCV gives us YCrCb internally — we re-order to match standard Y/Cb/Cr naming
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y  = ycrcb[:, :, 0]
    Cb = ycrcb[:, :, 2]   # OpenCV stores Cr second, Cb third
    Cr = ycrcb[:, :, 1]
    return Y, Cb, Cr


def subsample_420(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    4:2:0 chroma subsampling: keep Y at full resolution,
    downsample Cb and Cr by 2× in both horizontal and vertical directions.
    """
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]
    return Y, Cb_sub, Cr_sub


# ─── Decoder ──────────────────────────────────────────────────────────────────

def upsample_420(Y: np.ndarray, Cb_sub: np.ndarray, Cr_sub: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Upsample Cb and Cr back to full resolution using bilinear interpolation.
    """
    H, W = Y.shape
    Cb = cv2.resize(Cb_sub, (W, H), interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr_sub, (W, H), interpolation=cv2.INTER_LINEAR)
    return Y, Cb, Cr


def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Reconstruct a BGR uint8 image from Y, Cb, Cr float32 channels.
    """
    ycrcb = np.stack([Y, Cr, Cb], axis=2)   # OpenCV YCrCb order
    ycrcb = np.clip(ycrcb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return bgr
