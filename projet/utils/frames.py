"""
Utility: Frame I/O helpers
"""

import os
import glob
import cv2
import numpy as np


def load_frames(folder: str) -> list[np.ndarray]:
    """
    Load all PNG/JPG frames from a folder, sorted by filename.
    Returns a list of BGR uint8 numpy arrays.
    """
    paths = sorted(
        glob.glob(os.path.join(folder, '*.png')) +
        glob.glob(os.path.join(folder, '*.jpg')) +
        glob.glob(os.path.join(folder, '*.jpeg'))
    )
    if not paths:
        raise FileNotFoundError(f"No image frames found in: {folder}")
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise IOError(f"Could not read image: {p}")
        frames.append(img)
    return frames


def save_frames(frames: list[np.ndarray], folder: str,
                prefix: str = 'frame',
                fmt: str = 'png') -> None:
    """
    Save a list of BGR frames to a folder as PNG/JPG files.
    """
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames):
        path = os.path.join(folder, f'{prefix}_{i:04d}.{fmt}')
        cv2.imwrite(path, frame)
