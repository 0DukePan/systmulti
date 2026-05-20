"""
Stage 5 — Pipeline Visualisation
matplotlib-based visualizations for every stage of the MPEG-4 pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.fft import dctn
import cv2
import os


plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'text.color':       'white',
    'axes.labelcolor':  'white',
    'xtick.color':      '#aaaaaa',
    'ytick.color':      '#aaaaaa',
    'axes.titlecolor':  '#e67e22',
    'axes.edgecolor':   '#444',
    'savefig.facecolor': '#1a1a2e',
    'savefig.dpi': 150,
})

ACCENT = '#e67e22'
CMAP_Y  = 'gray'
CMAP_CB = 'Blues_r'
CMAP_CR = 'Reds_r'


def _title_frame(ax, title):
    ax.set_title(title, pad=6, weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])


# ─── 1. YCbCr Channel Decomposition ───────────────────────────────────────────

def plot_ycbcr_channels(frame_bgr: np.ndarray,
                        save_path: str = None) -> plt.Figure:
    """4-panel: original BGR + Y / Cb / Cr channels."""
    from pipeline.preprocessing import bgr_to_ycbcr, subsample_420

    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    _, Cb_sub, Cr_sub = subsample_420(Y, Cb, Cr)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Stage 1 — YCbCr Decomposition & 4:2:0 Subsampling',
                 color=ACCENT, fontsize=13, weight='bold')

    axes[0].imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    _title_frame(axes[0], 'Original (BGR)')

    axes[1].imshow(Y, cmap=CMAP_Y, vmin=0, vmax=255)
    _title_frame(axes[1], f'Y (Luminance)\n{Y.shape[1]}×{Y.shape[0]}')

    axes[2].imshow(Cb_sub, cmap=CMAP_CB, vmin=0, vmax=255)
    _title_frame(axes[2],
                 f'Cb (Blue-diff) — 4:2:0\n{Cb_sub.shape[1]}×{Cb_sub.shape[0]}')

    axes[3].imshow(Cr_sub, cmap=CMAP_CR, vmin=0, vmax=255)
    _title_frame(axes[3],
                 f'Cr (Red-diff) — 4:2:0\n{Cr_sub.shape[1]}×{Cr_sub.shape[0]}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ─── 2. DCT Pipeline (8×8 block) ──────────────────────────────────────────────

def plot_dct_pipeline(block_8x8: np.ndarray,
                      qf: int = 50,
                      save_path: str = None) -> plt.Figure:
    """4-panel: raw block → DCT → quantised → reconstructed."""
    from pipeline.intra import get_quant_matrix, _dct_block, _idct_block, _quantize, _dequantize

    block = block_8x8.astype(np.float32)
    dct   = _dct_block(block)
    qmat  = get_quant_matrix(qf)
    qblk  = _quantize(dct, qmat).astype(np.float32)
    dblk  = _dequantize(qblk.astype(np.int16), qmat)
    recon = np.clip(_idct_block(dblk), 0, 255)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Stage 2 — DCT Compression Pipeline  [QF={qf}]',
                 color=ACCENT, fontsize=13, weight='bold')

    subplots = [
        (block,  CMAP_Y,   'Raw 8×8 Block'),
        (np.log1p(np.abs(dct)), 'plasma', 'DCT Coefficients (log scale)'),
        (qblk,   'RdYlGn','Quantised Coefficients'),
        (recon,  CMAP_Y,   'Reconstructed Block'),
    ]
    for ax, (data, cmap, title) in zip(axes, subplots):
        im = ax.imshow(data, cmap=cmap, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _title_frame(ax, title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ─── 3. Motion Vectors on P-frame ─────────────────────────────────────────────

def plot_motion_vectors(frame_Y: np.ndarray,
                        motion_vectors: list,
                        differential: bool = True,
                        save_path: str = None) -> plt.Figure:
    """
    Draw absolute motion vector arrows overlaid on the P-frame.
    If differential=True, reconstructs absolute MVs from differential coding
    (cumulative sum) before plotting.
    """
    from pipeline.inter import _mv_from_differential

    # Reconstruct absolute MVs if stored as differential
    abs_mvs = _mv_from_differential(motion_vectors) if differential else motion_vectors

    H, W = frame_Y.shape
    cols = (W + 15) // 16
    rows = (H + 15) // 16

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Stage 3 — Motion Vectors on P-frame (Differential MV Coding)',
                 color=ACCENT, fontsize=13, weight='bold')
    ax.imshow(frame_Y, cmap=CMAP_Y, vmin=0, vmax=255)

    mv_idx = 0
    for r in range(rows):
        for c in range(cols):
            if mv_idx >= len(abs_mvs):
                break
            dy, dx = abs_mvs[mv_idx]
            y0 = r * 16 + 8
            x0 = c * 16 + 8
            if dy != 0 or dx != 0:
                ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='->', color=ACCENT,
                                            lw=1.5))
            mv_idx += 1

    # Draw macroblock grid
    for r in range(0, H, 16):
        ax.axhline(r, color='#444', lw=0.4)
    for c in range(0, W, 16):
        ax.axvline(c, color='#444', lw=0.4)

    magnitude = np.sqrt(np.array([d**2 + d2**2
                                   for d, d2 in abs_mvs], dtype=float))
    ax.set_title(f'Avg motion: {magnitude.mean():.1f}px  '
                 f'Max: {magnitude.max():.0f}px  '
                 f'(stored as differential deltas)',
                 color='#aaa', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ─── 4. Residuals ─────────────────────────────────────────────────────────────

def plot_residuals(residual: np.ndarray, reconstructed: np.ndarray,
                   original: np.ndarray = None,
                   save_path: str = None) -> plt.Figure:
    """Show residual signal and reconstructed P-frame."""
    n = 3 if original is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.suptitle('Stage 3 — Residuals & Reconstruction',
                 color=ACCENT, fontsize=13, weight='bold')

    axes[0].imshow(residual, cmap='bwr',
                   norm=Normalize(vmin=-60, vmax=60))
    _title_frame(axes[0], 'Residual (Motion-Compensated Error)')

    axes[1].imshow(reconstructed, cmap=CMAP_Y, vmin=0, vmax=255)
    _title_frame(axes[1], 'Reconstructed P-frame')

    if original is not None:
        axes[2].imshow(original, cmap=CMAP_Y, vmin=0, vmax=255)
        _title_frame(axes[2], 'Original frame')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ─── 5. Compression ratio vs QF ───────────────────────────────────────────────

def plot_compression_analysis(qf_list: list, ratios: list,
                               psnr_list: list = None,
                               save_path: str = None) -> plt.Figure:
    """Compression ratio and optional PSNR vs QF."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle('Experimental Analysis — Compression Ratio vs Quality Factor',
                 color=ACCENT, fontsize=13, weight='bold')

    color1 = '#3498db'
    ax1.plot(qf_list, ratios, 'o-', color=color1, lw=2, markersize=6,
             label='Compression Ratio')
    ax1.set_xlabel('Quality Factor (QF)')
    ax1.set_ylabel('Compression Ratio (×)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.2)

    if psnr_list:
        ax2 = ax1.twinx()
        color2 = '#e74c3c'
        ax2.plot(qf_list, psnr_list, 's--', color=color2, lw=2, markersize=6,
                 label='PSNR (dB)')
        ax2.set_ylabel('PSNR (dB)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                   framealpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ─── 6. PSNR vs GOP size ──────────────────────────────────────────────────────

def plot_gop_effect(gop_list: list, psnr_list: list, ratio_list: list = None,
                    save_path: str = None) -> plt.Figure:
    """PSNR vs GOP size plot."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle('Experimental Analysis — Effect of GOP Size',
                 color=ACCENT, fontsize=13, weight='bold')

    color1 = '#2ecc71'
    ax1.plot(gop_list, psnr_list, 'o-', color=color1, lw=2, markersize=6,
             label='Avg PSNR (dB)')
    ax1.set_xlabel('GOP Size (G)')
    ax1.set_ylabel('Avg PSNR (dB)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.2)

    if ratio_list:
        ax2 = ax1.twinx()
        color2 = '#9b59b6'
        ax2.plot(gop_list, ratio_list, 's--', color=color2, lw=2, markersize=6,
                 label='Compression Ratio')
        ax2.set_ylabel('Compression Ratio (×)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                   framealpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ─── 7. Temporal PSNR Curve ───────────────────────────────────────────────────

def plot_temporal_psnr(psnr_values: list,
                       frame_types_list: list,
                       gop: int = None,
                       save_path: str = None) -> plt.Figure:
    """
    Plot PSNR value for each frame, highlighting I-frames vs P-frames.

    This is the key experimental analysis plot: it shows how quality drops
    slightly on P-frames and recovers at each I-frame (start of new GOP).

    Parameters
    ----------
    psnr_values      : list of float, one PSNR per frame
    frame_types_list : list of 'I' or 'P', one per frame
    gop              : GOP size (for annotation)
    save_path        : optional output path
    """
    n = len(psnr_values)
    frames = list(range(n))

    i_frames = [i for i, t in enumerate(frame_types_list) if t == 'I']
    p_frames = [i for i, t in enumerate(frame_types_list) if t == 'P']

    i_psnr = [psnr_values[i] for i in i_frames]
    p_psnr = [psnr_values[i] for i in p_frames]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle('Temporal PSNR — Effect of GOP Structure on Frame Quality',
                 color=ACCENT, fontsize=13, weight='bold')

    # Background bands for each GOP
    if gop:
        for g in range(0, n, gop):
            ax.axvspan(g, min(g + gop, n), alpha=0.07,
                       color='#e67e22' if (g // gop) % 2 == 0 else '#3498db')

    # Full PSNR line
    ax.plot(frames, psnr_values, '-', color='#95a5a6', lw=1.2,
            alpha=0.6, label='PSNR per frame')

    # I-frames (green dots)
    ax.scatter(i_frames, i_psnr, color='#2ecc71', zorder=5, s=80,
               marker='D', label='I-frame (intra)')

    # P-frames (orange dots)
    ax.scatter(p_frames, p_psnr, color=ACCENT, zorder=4, s=40,
               marker='o', alpha=0.8, label='P-frame (inter)')

    # Vertical lines at I-frame positions
    for i in i_frames:
        ax.axvline(i, color='#2ecc71', lw=0.8, alpha=0.5, linestyle='--')

    avg = float(np.mean(psnr_values))
    ax.axhline(avg, color='white', lw=1, linestyle=':', alpha=0.4,
               label=f'Avg PSNR = {avg:.2f} dB')

    ax.set_xlabel('Frame number')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(True, alpha=0.15)
    ax.legend(loc='lower right', framealpha=0.3, fontsize=9)

    gop_label = f'  GOP = {gop}' if gop else ''
    ax.set_title(f'Avg: {avg:.2f} dB{gop_label}   '
                 f'I-frames: {len(i_frames)}   P-frames: {len(p_frames)}',
                 color='#aaaaaa', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ─── Full pipeline demo (all-in-one) ──────────────────────────────────────────

def demo_all(frame_dir: str = 'frames',
             out_dir: str = 'visualizations') -> None:
    """
    Generate all visualization panels using the first available frame.
    Outputs PNG files to out_dir/.
    """
    import glob
    from pipeline.intra import get_quant_matrix, _dct_block, _quantize

    os.makedirs(out_dir, exist_ok=True)
    frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.png')) +
                         glob.glob(os.path.join(frame_dir, '*.jpg')))
    if not frame_files:
        print(f"No frames found in '{frame_dir}'. Run generate_frames.py first.")
        return

    frame0 = cv2.imread(frame_files[0])
    frame1 = cv2.imread(frame_files[min(1, len(frame_files)-1)])

    from pipeline.preprocessing import bgr_to_ycbcr, subsample_420, upsample_420, ycbcr_to_bgr
    from pipeline.inter import encode_pframe

    # 1. YCbCr
    fig = plot_ycbcr_channels(frame0)
    fig.savefig(os.path.join(out_dir, '1_ycbcr_channels.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [1/6] YCbCr channels saved.")

    # 2. DCT block
    Y0, _, _ = bgr_to_ycbcr(frame0)
    block = Y0[16:24, 16:24]   # pick a representative 8×8 block
    fig = plot_dct_pipeline(block, qf=50)
    fig.savefig(os.path.join(out_dir, '2_dct_pipeline.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [2/6] DCT pipeline saved.")

    # 3. Motion vectors
    Y0f = Y0
    Y1, _, _ = bgr_to_ycbcr(frame1)
    pdata = encode_pframe(Y1, Y0f, S=8, qf=50)
    fig = plot_motion_vectors(Y1, pdata['mv'])
    fig.savefig(os.path.join(out_dir, '3_motion_vectors.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [3/6] Motion vectors saved.")

    # 4. Residuals
    from pipeline.inter import decode_pframe
    recon_Y = decode_pframe(pdata, Y0f)
    residual_vis = Y1[:recon_Y.shape[0], :recon_Y.shape[1]] - recon_Y
    fig = plot_residuals(residual_vis, recon_Y, Y1[:recon_Y.shape[0], :recon_Y.shape[1]])
    fig.savefig(os.path.join(out_dir, '4_residuals.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [4/6] Residuals saved.")

    # 5. Compression ratio vs QF (synthetic — quick demo)
    qf_vals = [5, 10, 20, 30, 50, 70, 90]
    synth_ratios = [18.2, 12.5, 8.1, 5.9, 3.7, 2.4, 1.6]
    synth_psnrs  = [22.1, 25.8, 29.3, 31.5, 34.2, 37.8, 41.2]
    fig = plot_compression_analysis(qf_vals, synth_ratios, synth_psnrs)
    fig.savefig(os.path.join(out_dir, '5_compression_vs_qf.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [5/6] Compression vs QF saved.")

    # 6. GOP effect (synthetic — quick demo)
    gop_vals    = [1, 2, 5, 10, 15, 20]
    gop_psnrs   = [34.2, 33.8, 33.1, 32.5, 31.8, 30.9]
    gop_ratios  = [3.7,  4.2,  5.1,  6.8,  7.9,  9.1]
    fig = plot_gop_effect(gop_vals, gop_psnrs, gop_ratios)
    fig.savefig(os.path.join(out_dir, '6_gop_effect.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [6/6] GOP effect saved.")

    print(f"\nAll visualizations saved to '{out_dir}/'")
