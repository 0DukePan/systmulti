"""
Stage 2 — Intra-frame Coding (I-frames)
DCT-based spatial compression on 8×8 blocks:
  Encode: split → DCT → quantize → zigzag → RLE
  Decode: de-RLE → inv-zigzag → dequantize → IDCT
"""

import numpy as np
from scipy.fft import dctn, idctn


# ─── Standard JPEG luminance quantization matrix ───────────────────────────────
LUMA_QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

CHROMA_QUANT_MATRIX = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)

# Zigzag scan order for 8×8 block
ZIGZAG_IDX = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
], dtype=np.int32)

# Inverse zigzag: position in flat array → (row, col)
_INV_ZIGZAG = np.zeros(64, dtype=np.int32)
for _i, _z in enumerate(ZIGZAG_IDX):
    _INV_ZIGZAG[_z] = _i


def get_quant_matrix(qf: int, chroma: bool = False) -> np.ndarray:
    """
    Scale the quantization matrix by quality factor QF (1–100).
    QF=50 → standard matrix; QF<50 → lower quality (more compression);
    QF>50 → higher quality (less compression).
    """
    base = CHROMA_QUANT_MATRIX if chroma else LUMA_QUANT_MATRIX
    if qf <= 0:
        qf = 1
    if qf >= 100:
        return np.ones_like(base, dtype=np.float32)
    scale = (5000 / qf) if qf < 50 else (200 - 2 * qf)
    scaled = np.floor((base * scale + 50) / 100).astype(np.float32)
    scaled = np.clip(scaled, 1, 255)
    return scaled


def _dct_block(block: np.ndarray) -> np.ndarray:
    """Apply 2-D DCT-II to a float32 8×8 block."""
    return dctn(block - 128.0, type=2, norm='ortho')


def _idct_block(block: np.ndarray) -> np.ndarray:
    """Apply 2-D inverse DCT to a float32 8×8 block."""
    return idctn(block, type=2, norm='ortho') + 128.0


def _quantize(dct_block: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    return np.round(dct_block / qmat).astype(np.int16)


def _dequantize(qblock: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    return (qblock.astype(np.float32)) * qmat


def _zigzag(block: np.ndarray) -> np.ndarray:
    """Flatten an 8×8 block in zigzag order."""
    flat = block.flatten()
    return flat[ZIGZAG_IDX]


def _inv_zigzag(vec: np.ndarray) -> np.ndarray:
    """Reconstruct 8×8 block from zigzag vector."""
    flat = np.zeros(64, dtype=vec.dtype)
    flat[ZIGZAG_IDX] = vec
    return flat.reshape(8, 8)


def _rle_encode(vec: np.ndarray) -> list:
    """
    Simple Run-Length Encoding for a zigzag-scanned block.
    Returns list of (zero_run_length, value) pairs; (0,0) = EOB.
    """
    result = []
    zero_count = 0
    for v in vec:
        if v == 0:
            zero_count += 1
        else:
            result.append((zero_count, int(v)))
            zero_count = 0
    result.append((0, 0))   # End-of-Block
    return result


def _rle_decode(rle: list, length: int = 64) -> np.ndarray:
    """Decode RLE back to a zigzag vector of given length."""
    vec = np.zeros(length, dtype=np.int16)
    idx = 0
    for (run, val) in rle:
        if val == 0 and run == 0:
            break   # EOB
        idx += run
        if idx < length:
            vec[idx] = val
            idx += 1
    return vec


# ─── Pad frame to multiple of 8 ───────────────────────────────────────────────

def _pad(channel: np.ndarray) -> tuple[np.ndarray, tuple]:
    H, W = channel.shape
    pH = (8 - H % 8) % 8
    pW = (8 - W % 8) % 8
    padded = np.pad(channel, ((0, pH), (0, pW)), mode='edge')
    return padded, (H, W)


# ─── Public encode / decode ────────────────────────────────────────────────────

def encode_intra(channel: np.ndarray, qf: int = 50,
                 chroma: bool = False) -> dict:
    """
    Encode a single YCbCr channel as an I-frame.

    Parameters
    ----------
    channel : 2-D float32 array
    qf      : quality factor (1–100)
    chroma  : use chroma quantization matrix if True

    Returns
    -------
    dict with keys: 'rle', 'shape', 'qf', 'chroma'
    """
    padded, orig_shape = _pad(channel)
    H, W = padded.shape
    qmat = get_quant_matrix(qf, chroma)
    rle_data = []

    for row in range(0, H, 8):
        for col in range(0, W, 8):
            block = padded[row:row+8, col:col+8].astype(np.float32)
            dct   = _dct_block(block)
            qblk  = _quantize(dct, qmat)
            zzed  = _zigzag(qblk)
            rle   = _rle_encode(zzed)
            rle_data.append(rle)

    return {'rle': rle_data, 'shape': orig_shape, 'qf': qf, 'chroma': chroma}


def decode_intra(data: dict) -> np.ndarray:
    """
    Decode an I-frame channel from its encoded data dict.

    Returns a float32 2-D array of the original shape.
    """
    orig_shape = data['shape']
    qf         = data['qf']
    chroma     = data['chroma']
    qmat       = get_quant_matrix(qf, chroma)

    H = (orig_shape[0] + 7) // 8 * 8
    W = (orig_shape[1] + 7) // 8 * 8
    reconstructed = np.zeros((H, W), dtype=np.float32)

    block_idx = 0
    for row in range(0, H, 8):
        for col in range(0, W, 8):
            rle   = data['rle'][block_idx]
            zzed  = _rle_decode(rle)
            qblk  = _inv_zigzag(zzed)
            dblk  = _dequantize(qblk, qmat)
            block = _idct_block(dblk)
            reconstructed[row:row+8, col:col+8] = block
            block_idx += 1

    return np.clip(reconstructed[:orig_shape[0], :orig_shape[1]], 0, 255)
