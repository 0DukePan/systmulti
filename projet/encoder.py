"""
MPEG-4 Simplified Pipeline — Encoder
=====================================
Usage:
  python3 encoder.py --input frames/ --output video.bin [--gop 10] [--qf 50] [--search 8]
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from utils.frames import load_frames
from utils.metrics import compression_ratio, psnr
from pipeline.preprocessing import bgr_to_ycbcr, subsample_420
from pipeline.intra import encode_intra, decode_intra
from pipeline.inter import encode_pframe, decode_pframe, frame_types
from pipeline.entropy import compress_stream


def encode(input_dir: str, output_path: str,
           gop: int = 10, qf: int = 50, search: int = 8,
           verbose: bool = True) -> dict:
    """
    Full encoding pipeline.

    Parameters
    ----------
    input_dir   : folder containing sequential PNG/JPG frames
    output_path : output .bin file
    gop         : Group-of-Pictures size (every gop-th frame = I-frame)
    qf          : Quality Factor 1–100
    search      : Block matching search window ±S pixels

    Returns
    -------
    dict with encoding statistics
    """
    t0 = time.time()

    # ── Load frames ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"  MPEG-4 Simplified Encoder")
        print(f"{'='*60}")
        print(f"  Input  : {input_dir}")
        print(f"  Output : {output_path}")
        print(f"  GOP    : {gop}   QF : {qf}   Search : ±{search}px")
        print(f"{'='*60}\n")

    frames = load_frames(input_dir)
    H, W = frames[0].shape[:2]
    n = len(frames)
    types = frame_types(n, gop)

    if verbose:
        print(f"  Loaded {n} frames ({W}×{H})")
        n_i = types.count('I')
        n_p = types.count('P')
        print(f"  Frame types: {n_i} I-frames, {n_p} P-frames")
        print()

    encoded_frames = []
    prev_Y = None   # previous reconstructed Y channel (for P-frame reference)

    with tqdm(total=n, desc="  Encoding", unit="frame", disable=not verbose) as pbar:
        for i, (frame, ftype) in enumerate(zip(frames, types)):

            # Stage 1 — Pre-processing
            Y, Cb, Cr = bgr_to_ycbcr(frame)
            Y_f, Cb_sub, Cr_sub = subsample_420(Y, Cb, Cr)

            if ftype == 'I':
                # Stage 2 — Intra coding
                Y_enc  = encode_intra(Y_f,    qf=qf, chroma=False)
                Cb_enc = encode_intra(Cb_sub, qf=qf, chroma=True)
                Cr_enc = encode_intra(Cr_sub, qf=qf, chroma=True)

                # Decode to get reference for next P-frame
                prev_Y = decode_intra(Y_enc)

                encoded_frames.append({
                    'type': 'I',
                    'Y':  Y_enc,
                    'Cb': Cb_enc,
                    'Cr': Cr_enc,
                })

            else:
                # Stage 3 — Inter coding
                Y_pdata = encode_pframe(Y_f, prev_Y, S=search, qf=qf)
                Cb_enc  = encode_intra(Cb_sub, qf=qf, chroma=True)
                Cr_enc  = encode_intra(Cr_sub, qf=qf, chroma=True)

                # Decode Y to use as reference for next frame
                prev_Y = decode_pframe(Y_pdata, prev_Y)

                encoded_frames.append({
                    'type': 'P',
                    'Y':  Y_pdata,
                    'Cb': Cb_enc,
                    'Cr': Cr_enc,
                })

            pbar.update(1)

    # Stage 4 — Entropy coding
    if verbose:
        print(f"\n  Writing compressed .bin file...")

    bin_size = compress_stream(encoded_frames, output_path,
                               width=W, height=H, gop=gop, qf=qf)

    # Compute raw size (3 bytes/pixel × W × H × n_frames)
    raw_size = W * H * 3 * n
    stats = compression_ratio(raw_size, bin_size)
    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Encoding complete in {elapsed:.2f}s")
        print(f"  Raw size       : {raw_size / 1024:.1f} KB")
        print(f"  Compressed     : {bin_size / 1024:.1f} KB")
        print(f"  Ratio          : {stats['ratio']:.2f}×")
        print(f"  Space saved    : {stats['space_saved']:.1f}%")
        print(f"{'='*60}\n")

    return {
        'n_frames':   n,
        'width':      W,
        'height':     H,
        'gop':        gop,
        'qf':         qf,
        'search':     search,
        'raw_bytes':  raw_size,
        'bin_bytes':  bin_size,
        'ratio':      stats['ratio'],
        'space_saved': stats['space_saved'],
        'elapsed':    elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description='MPEG-4 Simplified Encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',  '-i', required=True,
                        help='Input folder containing sequential image frames')
    parser.add_argument('--output', '-o', default='video.bin',
                        help='Output .bin file')
    parser.add_argument('--gop',    '-g', type=int, default=10,
                        help='Group-of-Pictures size')
    parser.add_argument('--qf',     '-q', type=int, default=50,
                        help='Quality Factor (1–100, lower=more compression)')
    parser.add_argument('--search', '-s', type=int, default=8,
                        help='Block matching search range ±S pixels')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: input folder '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    encode(args.input, args.output,
           gop=args.gop, qf=args.qf, search=args.search)


if __name__ == '__main__':
    main()
