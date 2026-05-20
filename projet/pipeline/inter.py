"""
Stage 3 — Inter-frame Coding (P-frames)
GOP structure + block matching + DIFFERENTIAL motion vector coding + residual.

Differential MV coding:
  Instead of storing absolute (dy, dx) for each macroblock,
  we store the delta relative to the previous macroblock's MV.
  Adjacent blocks usually move in the same direction → deltas ≈ 0 → better Huffman/zlib compression.
"""

import numpy as np
from pipeline.intra import encode_intra, decode_intra, _pad


# ─── Block matching ────────────────────────────────────────────────────────────

def _sad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _block_match(curr_block: np.ndarray, ref: np.ndarray,
                 row: int, col: int, S: int) -> tuple[int, int]:
    """
    Full-search block matching for a 16×16 macroblock within ±S pixels.
    Returns (dy, dx) — the best motion vector (absolute).
    """
    H, W = ref.shape
    bh, bw = curr_block.shape
    best_sad = float('inf')
    best_dy, best_dx = 0, 0

    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            r = row + dy
            c = col + dx
            if r < 0 or c < 0 or r + bh > H or c + bw > W:
                continue
            s = _sad(curr_block, ref[r:r+bh, c:c+bw])
            if s < best_sad:
                best_sad = s
                best_dy, best_dx = dy, dx

    return best_dy, best_dx


# ─── Differential MV coding ───────────────────────────────────────────────────

def _mv_to_differential(mv_list: list[tuple]) -> list[tuple]:
    """
    Convert absolute motion vectors to differential (predictive) coding.
    Each MV is stored as: delta = MV[i] - MV[i-1].
    MV[0] is stored as-is (reference = (0,0)).

    This exploits the fact that adjacent macroblocks usually move similarly,
    so deltas are small → better entropy compression.
    """
    if not mv_list:
        return []
    diff = [mv_list[0]]
    for i in range(1, len(mv_list)):
        dy = mv_list[i][0] - mv_list[i-1][0]
        dx = mv_list[i][1] - mv_list[i-1][1]
        diff.append((dy, dx))
    return diff


def _mv_from_differential(diff_list: list[tuple]) -> list[tuple]:
    """Reconstruct absolute MVs from differential list (cumulative sum)."""
    if not diff_list:
        return []
    mv = [diff_list[0]]
    for i in range(1, len(diff_list)):
        dy = mv[-1][0] + diff_list[i][0]
        dx = mv[-1][1] + diff_list[i][1]
        mv.append((dy, dx))
    return mv


# ─── GOP structure ─────────────────────────────────────────────────────────────

def frame_types(n_frames: int, gop: int) -> list[str]:
    """Return list of 'I' or 'P' for each frame index."""
    return ['I' if i % gop == 0 else 'P' for i in range(n_frames)]


# ─── Encoder ──────────────────────────────────────────────────────────────────

def encode_pframe(curr_Y: np.ndarray, ref_Y: np.ndarray,
                  S: int = 8, qf: int = 50) -> dict:
    """
    Encode a P-frame Y channel using motion-compensated prediction
    with differential MV coding.

    Parameters
    ----------
    curr_Y : current frame Y (float32, H×W)
    ref_Y  : previous reconstructed Y (float32, H×W)
    S      : block-matching search radius (pixels)
    qf     : quality factor for residual DCT coding

    Returns
    -------
    dict with keys: 'mv' (differential), 'residual_encoded', 'shape', 'S', 'qf'
    """
    curr_pad, orig_shape = _pad(curr_Y)
    ref_pad, _           = _pad(ref_Y)
    H, W = curr_pad.shape

    abs_mvs = []
    mc_frame = np.zeros_like(curr_pad)

    for row in range(0, H, 16):
        for col in range(0, W, 16):
            r_end = min(row + 16, H)
            c_end = min(col + 16, W)
            bh, bw = r_end - row, c_end - col
            curr_block = curr_pad[row:r_end, col:c_end].astype(np.float32)

            if bh == 16 and bw == 16:
                dy, dx = _block_match(curr_block, ref_pad, row, col, S)
            else:
                dy, dx = 0, 0   # border blocks — no search

            abs_mvs.append((int(dy), int(dx)))

            # Motion-compensated prediction block
            r_ref = int(np.clip(row + dy, 0, ref_pad.shape[0] - bh))
            c_ref = int(np.clip(col + dx, 0, ref_pad.shape[1] - bw))
            mc_frame[row:r_end, col:c_end] = ref_pad[r_ref:r_ref+bh, c_ref:c_ref+bw]

    # ── Differential MV coding ──────────────────────────────────────────────
    diff_mvs = _mv_to_differential(abs_mvs)

    # ── Residual = current − MC prediction ──────────────────────────────────
    residual = curr_pad.astype(np.float32) - mc_frame.astype(np.float32)
    residual_shifted = np.clip(residual + 128.0, 0, 255)

    residual_encoded = encode_intra(residual_shifted, qf=qf, chroma=False)

    return {
        'mv':               diff_mvs,   # differential MVs
        'residual_encoded': residual_encoded,
        'shape':            orig_shape,
        'S':                S,
        'qf':               qf,
    }


# ─── Decoder ──────────────────────────────────────────────────────────────────

def decode_pframe(data: dict, ref_Y: np.ndarray) -> np.ndarray:
    """
    Decode a P-frame Y channel.
    Reconstructs absolute MVs from differential, then applies
    motion compensation + residual addition.
    """
    orig_shape   = data['shape']
    diff_mvs     = data['mv']

    # Reconstruct absolute MVs from differential
    abs_mvs = _mv_from_differential(diff_mvs)

    ref_pad, _ = _pad(ref_Y)
    H_pad = (orig_shape[0] + 15) // 16 * 16
    W_pad = (orig_shape[1] + 15) // 16 * 16

    rH, rW = ref_pad.shape
    if rH < H_pad or rW < W_pad:
        ref_pad = np.pad(ref_pad,
                         ((0, max(0, H_pad - rH)),
                          (0, max(0, W_pad - rW))),
                         mode='edge')

    mc_frame = np.zeros((H_pad, W_pad), dtype=np.float32)
    mv_idx = 0
    for row in range(0, H_pad, 16):
        for col in range(0, W_pad, 16):
            r_end = min(row + 16, H_pad)
            c_end = min(col + 16, W_pad)
            bh, bw = r_end - row, c_end - col

            if mv_idx < len(abs_mvs):
                dy, dx = abs_mvs[mv_idx]
                mv_idx += 1
            else:
                dy, dx = 0, 0

            r_ref = int(np.clip(row + dy, 0, ref_pad.shape[0] - bh))
            c_ref = int(np.clip(col + dx, 0, ref_pad.shape[1] - bw))
            mc_frame[row:r_end, col:c_end] = ref_pad[r_ref:r_ref+bh, c_ref:c_ref+bw]

    # Decode residual
    H_rd = (orig_shape[0] + 7) // 8 * 8
    W_rd = (orig_shape[1] + 7) // 8 * 8
    residual_shifted = decode_intra(data['residual_encoded'])
    residual = residual_shifted[:H_rd, :W_rd] - 128.0

    recon = mc_frame[:H_rd, :W_rd] + residual
    return np.clip(recon[:orig_shape[0], :orig_shape[1]], 0, 255)
