"""
MPEG-4 Simplified Pipeline — Decoder
=====================================
Usage:
  python3 decoder.py --input video.bin --output decoded/
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2
from tqdm import tqdm

from utils.frames import save_frames
from pipeline.preprocessing import upsample_420, ycbcr_to_bgr
from pipeline.intra import decode_intra
from pipeline.inter import decode_pframe
from pipeline.entropy import decompress_stream


def decode(bin_path: str, output_dir: str, verbose: bool = True) -> dict:
    """
    Full decoding pipeline.

    Parameters
    ----------
    bin_path   : input .bin file (produced by encoder)
    output_dir : folder where decoded PNG frames will be saved

    Returns
    -------
    dict with decoding statistics
    """
    t0 = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MPEG-4 Simplified Decoder")
        print(f"{'='*60}")
        print(f"  Input  : {bin_path}")
        print(f"  Output : {output_dir}")
        print(f"{'='*60}\n")

    # Stage 4 — Decompress bitstream
    encoded_frames, header = decompress_stream(bin_path)
    W, H   = header['width'], header['height']
    n      = header['n_frames']
    gop    = header['gop']
    qf     = header['qf']

    if verbose:
        print(f"  Header: {n} frames ({W}×{H}), GOP={gop}, QF={qf}")
        print()

    os.makedirs(output_dir, exist_ok=True)
    prev_Y = None

    with tqdm(total=n, desc="  Decoding", unit="frame", disable=not verbose) as pbar:
        for i, fdata in enumerate(encoded_frames):
            ftype = fdata['type']

            # ── Decode Y channel ──────────────────────────────────────────────
            if ftype == 'I':
                Y_dec = decode_intra(fdata['Y'])
            else:
                Y_dec = decode_pframe(fdata['Y'], prev_Y)

            prev_Y = Y_dec.copy()

            # ── Decode and upsample chroma ────────────────────────────────────
            Cb_dec = decode_intra(fdata['Cb'])
            Cr_dec = decode_intra(fdata['Cr'])
            Y_full, Cb_up, Cr_up = upsample_420(Y_dec, Cb_dec, Cr_dec)

            # ── Stage 1 inverse: YCbCr → BGR ─────────────────────────────────
            frame = ycbcr_to_bgr(Y_full, Cb_up, Cr_up)

            # ── Save decoded frame ────────────────────────────────────────────
            out_path = os.path.join(output_dir, f'frame_{i:04d}.png')
            cv2.imwrite(out_path, frame)
            pbar.update(1)

    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Decoding complete in {elapsed:.2f}s")
        print(f"  {n} frames saved to '{output_dir}/'")
        print(f"{'='*60}\n")

    return {
        'n_frames': n,
        'width':    W,
        'height':   H,
        'elapsed':  elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description='MPEG-4 Simplified Decoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',  '-i', required=True,
                        help='Input .bin file')
    parser.add_argument('--output', '-o', default='decoded',
                        help='Output folder for decoded frames')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    decode(args.input, args.output)


if __name__ == '__main__':
    main()
