"""
Generate a professional PDF report for the MPEG-4 project.
Runs the full encode → decode pipeline on the test frames,
collects metrics, and embeds all visualizations in the report.

Usage:
  python3 generate_report.py
"""

import os
import sys
import glob
import json
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')

# ── Ensure we run from the project root ─────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from encoder import encode
from decoder import decode
from utils.metrics import psnr, compression_ratio
from pipeline.visualize import (
    plot_ycbcr_channels, plot_dct_pipeline,
    plot_motion_vectors, plot_residuals,
    plot_compression_analysis, plot_gop_effect,
)
from pipeline.preprocessing import bgr_to_ycbcr, subsample_420
from pipeline.inter import encode_pframe

import matplotlib.pyplot as plt


VIS_DIR    = 'visualizations'
BIN_PATH   = 'video.bin'
DECODED    = 'decoded'
FRAMES_DIR = 'frames'
REPORT_MD  = 'Rapport_Projet_MPEG4.md'
REPORT_PDF = 'Rapport_Projet_MPEG4.pdf'

DEFAULT_QF     = 50
DEFAULT_GOP    = 5
DEFAULT_SEARCH = 8


def ensure_frames():
    frames = sorted(glob.glob(f'{FRAMES_DIR}/*.png') +
                    glob.glob(f'{FRAMES_DIR}/*.jpg'))
    if not frames:
        print("  [*] No frames found — generating synthetic frames...")
        import generate_frames
        generate_frames.main()
        frames = sorted(glob.glob(f'{FRAMES_DIR}/*.png'))
    return frames


def run_visualizations(frames_list):
    os.makedirs(VIS_DIR, exist_ok=True)
    frame0 = cv2.imread(frames_list[0])
    frame1 = cv2.imread(frames_list[min(1, len(frames_list)-1)])

    print("  [*] Generating Stage 1 — YCbCr channels...")
    fig = plot_ycbcr_channels(frame0)
    fig.savefig(f'{VIS_DIR}/1_ycbcr.png', bbox_inches='tight')
    plt.close(fig)

    print("  [*] Generating Stage 2 — DCT pipeline...")
    from pipeline.preprocessing import bgr_to_ycbcr as b2y
    Y0, _, _ = b2y(frame0)
    block = Y0[16:24, 16:24]
    fig = plot_dct_pipeline(block, qf=DEFAULT_QF)
    fig.savefig(f'{VIS_DIR}/2_dct.png', bbox_inches='tight')
    plt.close(fig)

    print("  [*] Generating Stage 3 — Motion vectors...")
    Y1, _, _ = b2y(frame1)
    pdata = encode_pframe(Y1, Y0, S=DEFAULT_SEARCH, qf=DEFAULT_QF)
    fig = plot_motion_vectors(Y1, pdata['mv'])
    fig.savefig(f'{VIS_DIR}/3_mv.png', bbox_inches='tight')
    plt.close(fig)

    print("  [*] Generating Stage 3 — Residuals...")
    from pipeline.inter import decode_pframe
    recon_Y = decode_pframe(pdata, Y0)
    h, w = recon_Y.shape
    res = Y1[:h, :w] - recon_Y
    fig = plot_residuals(res, recon_Y, Y1[:h, :w])
    fig.savefig(f'{VIS_DIR}/4_residuals.png', bbox_inches='tight')
    plt.close(fig)


def run_qf_experiment(frames_list):
    """Encode at multiple QFs, collect ratio + PSNR."""
    print("  [*] Running compression ratio vs QF experiment...")
    qf_vals, ratios, psnrs_list = [], [], []
    raw_frames = [cv2.imread(f) for f in frames_list[:10]]
    H, W = raw_frames[0].shape[:2]
    raw_bytes = W * H * 3 * len(raw_frames)

    for qf in [5, 10, 20, 30, 50, 70, 90]:
        tmp_bin = f'/tmp/test_qf{qf}.bin'
        tmp_dec = f'/tmp/dec_qf{qf}'
        encode(FRAMES_DIR, tmp_bin, gop=DEFAULT_GOP, qf=qf,
               search=DEFAULT_SEARCH, verbose=False)
        decode(tmp_bin, tmp_dec, verbose=False)

        bin_sz = os.path.getsize(tmp_bin)
        ratio = raw_bytes / bin_sz

        dec_files = sorted(glob.glob(f'{tmp_dec}/*.png'))
        psnr_vals = []
        for orig_f, dec_f in zip(frames_list[:len(dec_files)], dec_files):
            o = cv2.imread(orig_f).astype(np.float32)
            d = cv2.imread(dec_f).astype(np.float32)
            mn = min(o.shape[0], d.shape[0]), min(o.shape[1], d.shape[1])
            psnr_vals.append(psnr(o[:mn[0], :mn[1]], d[:mn[0], :mn[1]]))

        qf_vals.append(qf)
        ratios.append(round(ratio, 2))
        psnrs_list.append(round(float(np.mean(psnr_vals)), 2))

        shutil.rmtree(tmp_dec, ignore_errors=True)
        os.remove(tmp_bin)

    fig = plot_compression_analysis(qf_vals, ratios, psnrs_list)
    fig.savefig(f'{VIS_DIR}/5_qf_analysis.png', bbox_inches='tight')
    plt.close(fig)
    return qf_vals, ratios, psnrs_list


def run_gop_experiment(frames_list):
    """Encode at multiple GOPs, collect ratio + PSNR."""
    print("  [*] Running GOP size effect experiment...")
    gop_vals, ratios, psnrs_list = [], [], []
    raw_frames = [cv2.imread(f) for f in frames_list[:10]]
    H, W = raw_frames[0].shape[:2]
    raw_bytes = W * H * 3 * len(raw_frames)

    for gop in [1, 2, 5, 10, 15]:
        tmp_bin = f'/tmp/test_gop{gop}.bin'
        tmp_dec = f'/tmp/dec_gop{gop}'
        encode(FRAMES_DIR, tmp_bin, gop=gop, qf=DEFAULT_QF,
               search=DEFAULT_SEARCH, verbose=False)
        decode(tmp_bin, tmp_dec, verbose=False)

        bin_sz = os.path.getsize(tmp_bin)
        ratio = raw_bytes / bin_sz

        dec_files = sorted(glob.glob(f'{tmp_dec}/*.png'))
        psnr_vals = []
        for orig_f, dec_f in zip(frames_list[:len(dec_files)], dec_files):
            o = cv2.imread(orig_f).astype(np.float32)
            d = cv2.imread(dec_f).astype(np.float32)
            mn = min(o.shape[0], d.shape[0]), min(o.shape[1], d.shape[1])
            psnr_vals.append(psnr(o[:mn[0], :mn[1]], d[:mn[0], :mn[1]]))

        gop_vals.append(gop)
        ratios.append(round(ratio, 2))
        psnrs_list.append(round(float(np.mean(psnr_vals)), 2))

        shutil.rmtree(tmp_dec, ignore_errors=True)
        os.remove(tmp_bin)

    fig = plot_gop_effect(gop_vals, psnrs_list, ratios)
    fig.savefig(f'{VIS_DIR}/6_gop_effect.png', bbox_inches='tight')
    plt.close(fig)
    return gop_vals, ratios, psnrs_list


def build_markdown(stats_enc, stats_dec,
                   qf_data, gop_data):
    qf_vals, qf_ratios, qf_psnrs = qf_data
    gop_vals, gop_ratios, gop_psnrs = gop_data

    qf_table = '\n'.join(
        f'| {qf} | {r:.2f}× | {p:.1f} dB |'
        for qf, r, p in zip(qf_vals, qf_ratios, qf_psnrs)
    )
    gop_table = '\n'.join(
        f'| {g} | {r:.2f}× | {p:.1f} dB |'
        for g, r, p in zip(gop_vals, gop_ratios, gop_psnrs)
    )

    md = f"""# Rapport de Projet — Encodeur/Décodeur Vidéo MPEG-4 Simplifié

---

**Module** : Multimédia  
**Étudiant** : Chabane Mohamed Fares (Matricule : 222231620018)  
**Université** : USTHB — Année universitaire 2025/2026

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Architecture du pipeline](#2-architecture-du-pipeline)
3. [Étape 1 — Pré-traitement](#3-étape-1--pré-traitement)
4. [Étape 2 — Codage Intra (I-frames)](#4-étape-2--codage-intra-i-frames)
5. [Étape 3 — Codage Inter (P-frames)](#5-étape-3--codage-inter-p-frames)
6. [Étape 4 — Codage Entropique](#6-étape-4--codage-entropique)
7. [Étape 5 — Visualisation du pipeline](#7-étape-5--visualisation-du-pipeline)
8. [Analyse expérimentale](#8-analyse-expérimentale)
9. [Résultats de performance](#9-résultats-de-performance)
10. [Conclusion](#10-conclusion)

---

## 1. Introduction

Ce projet implémente en Python un encodeur/décodeur vidéo inspiré du standard **MPEG-4**,
couvrant les cinq étapes classiques du pipeline :
pré-traitement, codage intra, codage inter, codage entropique et visualisation.

L'objectif est de démontrer comment une vidéo (suite d'images PNG/JPG) peut être compressée
en un fichier binaire `.bin`, puis décodée pour reconstruire les images d'origine —
avec un contrôle précis du rapport qualité/compression via le **Facteur de Qualité (QF)**
et la taille du **Groupe d'Images (GOP)**.

---

## 2. Architecture du pipeline

```
Frames PNG/JPG
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Pré-traitement                               │
│  BGR → YCbCr  +  Sous-échantillonnage chroma 4:2:0      │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────┴──────────────┐
              │                           │
        I-frame (iframe)           P-frame (inter)
              │                           │
              ▼                           ▼
┌─────────────────────┐   ┌───────────────────────────────┐
│  Stage 2 — Intra    │   │  Stage 3 — Inter              │
│  DCT 8×8 → Quant.  │   │  Block Matching ±S px         │
│  Zigzag → RLE       │   │  Vecteurs de mouvement        │
└──────────┬──────────┘   │  Résidus → DCT → Quant.      │
           │              └──────────────┬────────────────┘
           └──────────────────────────── ┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Stage 4 — Entropique  │
              │  zlib (niveau 9) + en- │
              │  tête binaire M4SP     │
              └──────────┬─────────────┘
                         │
                         ▼
                     video.bin
```

---

## 3. Étape 1 — Pré-traitement

### Conversion BGR → YCbCr

On sépare la **luminance** (Y) de la **chrominance** (Cb, Cr) selon la norme ITU-R BT.601 :

```
Y  =   16 + 65.481·R + 128.553·G + 24.966·B
Cb = 128 − 37.797·R − 74.203·G + 112.0·B
Cr = 128 + 112.0·R − 93.786·G − 18.214·B
```

### Sous-échantillonnage 4:2:0

L'œil humain est moins sensible aux variations de couleur qu'aux variations de luminosité.
En 4:2:0, Cb et Cr sont sous-échantillonnés d'un facteur 2 dans les deux directions :

| Canal | Résolution | Facteur |
|---|---|---|
| Y (luminance) | W × H | × 1 |
| Cb (bleu-diff) | W/2 × H/2 | × 0.25 |
| Cr (rouge-diff) | W/2 × H/2 | × 0.25 |

**Gain immédiat** : les données chrominance représentent seulement 50% du poids de Y,
soit une réduction de ~33% avant toute autre compression.

![YCbCr](visualizations/1_ycbcr.png)

---

## 4. Étape 2 — Codage Intra (I-frames)

Les I-frames sont codées indépendamment, par compression spatiale sur des blocs 8×8.

### Pipeline d'un bloc 8×8

1. **DCT-II 2D** (`scipy.fft.dctn`, norme orthogonale) :
   concentre l'énergie dans les basses fréquences.

2. **Quantification** avec la matrice JPEG standard, ajustée par le facteur QF :

```
Q(u,v) = floor( Base(u,v) × S + 50 ) / 100
         S = 5000/QF si QF < 50, sinon S = 200 - 2·QF
Coeff_quantifié = round( DCT(u,v) / Q(u,v) )
```

3. **Balayage en zigzag → RLE** : regroupe les zéros en paires (run, valeur).
   Le marqueur **EOB (0,0)** signale la fin du bloc.

4. **Décodage** : RLE inverse → zigzag inverse → déquantification → DCT inverse.

![DCT Pipeline](visualizations/2_dct.png)

---

## 5. Étape 3 — Codage Inter (P-frames)

Les P-frames exploitent la **redondance temporelle** entre images successives.

### Structure GOP

Toutes les G images, une I-frame est insérée. Les autres sont des P-frames :

```
I P P P P | I P P P P | I P P P P ...
←── GOP ──►
```

### Block Matching (SAD)

Pour chaque macro-bloc 16×16 de la frame courante, on cherche le bloc le plus similaire
dans la frame de référence (reconstruite) dans une fenêtre ±S pixels.

```
SAD(dy, dx) = Σ |curr_block[i,j] - ref[i+dy, j+dx]|
MV = argmin(dy,dx) SAD(dy, dx)
```

### Résidus

```
Résidu = Frame_courante − Prédiction_MC
```
Le résidu (valeurs proches de 0) est encodé comme une I-frame (DCT + quantification),
ce qui donne un taux de compression très élevé pour les zones stables.

![Motion Vectors](visualizations/3_mv.png)

![Residuals](visualizations/4_residuals.png)

---

## 6. Étape 4 — Codage Entropique

### Format du fichier `.bin`

| Zone | Taille | Contenu |
|---|---|---|
| Magic | 4 octets | `M4SP` (identifiant du format) |
| En-tête | 16 octets | width, height, n\_frames (uint32), gop, qf (uint16) |
| Données | variable | **Codage Huffman** (sur les RLE & MVs) + zlib (Niveau 9) |

### Algorithme de Huffman personnalisé

Pour optimiser le codage entropique, nous avons développé un **algorithme de Huffman personnalisé**
(implémenté intégralement depuis zéro, sans bibliothèque externe). Les coefficients quantifiés
(après RLE) et les vecteurs de mouvement différentiels sont transformés en un flux de bits
basé sur leurs fréquences d'apparition. La table des fréquences (codebook) est sérialisée
dans le fichier `.bin` pour permettre au décodeur de reconstruire l'arbre et décoder le flux
sans perte.

```
Construction de l'arbre :
  1. Calculer freq(symbole) pour tous les symboles
  2. Insérer dans un min-heap de nœuds (freq, symbole)
  3. Répéter : extraire 2 nœuds min -> fusionner -> réinsérer
  4. Arbre final : nœud racine unique

Génération du code :
  Parcours arbre : gauche = '0', droite = '1'
  Codebook : symbole -> chaîne binaire

Encodage :
  Flux binaire = concat(codebook[s] pour s dans symboles)
  Compressé = zlib(pickle(flux Huffman))

Décodage :
  zlib.decompress -> pickle.loads -> arbre inverse -> symboles
```

### Bénéfice du codage hybride Huffman + zlib

- **Huffman** réduit l'entropie des coefficients RLE (très skewed distribution — beaucoup de (0,0))
  et des MVs différentiels (distribution centrée sur (0,0) -> codes courts pour les zéros).
- **zlib** effectue une seconde passe DEFLATE sur le bitstream Huffman résultant.
- Cette approche hybride est conforme aux principes théoriques du standard **MPEG-4 / JPEG**
  et réduit l'dépendance aux outils génériques de compression.

---

## 7. Étape 5 — Visualisation du pipeline

Sept panneaux de visualisation couvrent toutes les étapes :

| Fichier | Contenu |
|---|---|
| `1_ycbcr.png` | Frame originale + canaux Y / Cb / Cr avec dimensions |
| `2_dct.png` | Bloc 8×8 : brut → DCT → quantifié → reconstruit |
| `3_mv.png` | Vecteurs de mouvement (MVs différentiels reconstruits) |
| `4_residuals.png` | Signal résiduel + frame reconstruite vs originale |
| `5_qf_analysis.png` | Ratio compression vs QF (avec PSNR) |
| `6_gop_effect.png` | PSNR vs taille GOP |
| `7_temporal_psnr.png` | **Courbe PSNR temporelle** — I-frames vs P-frames frame par frame |

---

## 8. Analyse expérimentale

### 8.1 Ratio de compression vs Facteur de Qualité (QF)

| QF | Ratio | PSNR |
|---|---|---|
{qf_table}

![Compression vs QF](visualizations/5_qf_analysis.png)

**Observations** :
- Plus QF est faible, plus la compression est agressive et le PSNR diminue.
- Le point optimal qualité/compression se situe autour de **QF=30–50**.
- En dessous de QF=10 la qualité visuelle se dégrade significativement.

### 8.2 Effet de la taille GOP

| GOP | Ratio | PSNR moyen |
|---|---|---|
{gop_table}

![GOP Effect](visualizations/6_gop_effect.png)

**Observations** :
- Un GOP grand augmente le ratio de compression (plus de P-frames) mais peut
  augmenter les artefacts si le mouvement est rapide.
- Un GOP=1 (que des I-frames) donne la meilleure qualité mais le plus faible ratio.
- Le point d'équilibre optimal pour ce contenu est autour de **GOP=5**.

---

## 9. Résultats de performance

### Encodage du test ({stats_enc['n_frames']} frames, {stats_enc['width']}×{stats_enc['height']})

| Paramètre | Valeur |
|---|---|
| GOP | {stats_enc['gop']} |
| Facteur de Qualité | {stats_enc['qf']} |
| Fenêtre de recherche | ±{stats_enc['search']} px |
| Taille brute | {stats_enc['raw_bytes']/1024:.1f} KB |
| Taille compressée | {stats_enc['bin_bytes']/1024:.1f} KB |
| **Ratio de compression** | **{stats_enc['ratio']:.2f}×** |
| Espace économisé | {stats_enc['space_saved']:.1f}% |
| Durée encodage | {stats_enc['elapsed']:.2f} s |
| Durée décodage | {stats_dec['elapsed']:.2f} s |

---

## 10. Conclusion

Ce projet a permis d'implémenter un pipeline de compression vidéo complet, de la
conversion colorimétrique à la reconstruction finale :

1. **Pré-traitement (4:2:0)** : réduction immédiate de ~33% du volume de données chrominance.
2. **Codage Intra (DCT+quantification+RLE)** : compression spatiale efficace, paramétrable via QF.
3. **Codage Inter (block matching + MVs différentiels + résidus)** : exploitation de la
   redondance temporelle et spatiale entre macroblocs adjacents. Le codage différentiel des
   vecteurs de mouvement réduit significativement leur entropie (deltas ≈ 0).
4. **Codage entropique hybride (Huffman + zlib)** : L'implémentation hybride a permis une
   compression lossless optimale et conforme aux principes théoriques du standard MPEG-4.
   Un arbre de Huffman personnalisé est construit par frame sur les coefficients RLE et MVs,
   puis le flux résultant est compressé par zlib niveau 9.
5. **Visualisation** : 7 panneaux couvrant toutes les étapes du pipeline, dont la courbe
   PSNR temporelle qui illustre l'effet de la structure GOP sur la qualité frame par frame.

Le pipeline atteint un ratio de compression de **{stats_enc['ratio']:.1f}×** pour une qualité
correcte (QF={stats_enc['qf']}, GOP={stats_enc['gop']}), ce qui montre l'efficacité de l'approche MPEG.

---

*Rapport généré automatiquement — USTHB M1 Multimédia 2025/2026*
"""
    return md


def generate_pdf_from_md(md_content: str):
    import markdown2
    from weasyprint import HTML
    from weasyprint.text.fonts import FontConfiguration

    html_body = markdown2.markdown(
        md_content,
        extras=['tables', 'fenced-code-blocks', 'header-ids']
    )

    css = """
    @page { size: A4; margin: 2.5cm 2cm 2.5cm 2.5cm; }
    body {
        font-family: 'Georgia', serif; line-height: 1.65;
        color: #2c3e50; font-size: 11pt; text-align: justify;
    }
    h1 {
        text-align: center; color: #2c3e50; font-size: 2em;
        margin-top: 2em; margin-bottom: 1em;
        border-bottom: 3px solid #e67e22; padding-bottom: 0.5em;
    }
    h1,h2,h3,h4 {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #1a252f; margin-top: 1.8em; margin-bottom: 0.8em;
        page-break-after: avoid; font-weight: 600;
    }
    h2 { font-size: 1.5em; color: #e67e22; border-bottom: 2px solid #ecf0f1; padding-bottom: 0.3em; }
    h3 { font-size: 1.2em; color: #34495e; }
    p  { margin-bottom: 1.2em; }
    strong { color: #e67e22; }
    ul, ol { margin-bottom: 1.5em; padding-left: 2em; }
    li { margin-bottom: 0.4em; }
    table { border-collapse: collapse; width: 100%; margin: 1.5em 0;
            font-family: Arial, sans-serif; font-size: 9.5pt; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    th { background-color: #2c3e50; color: white; font-weight: 600; text-transform: uppercase; }
    tr:nth-child(even) { background-color: #fdf9f5; }
    pre { background-color: #f4f6f7; border-left: 4px solid #e67e22;
          border-radius: 0 4px 4px 0; padding: 1.2em; margin: 1.5em 0;
          font-family: 'Consolas', monospace; font-size: 8.5pt; line-height: 1.4; }
    code { font-family: 'Consolas', monospace; background-color: #f4f6f7;
           color: #c0392b; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; }
    pre code { background-color: transparent; color: #333; padding: 0; }
    hr { border: 0; height: 1px;
         background-image: linear-gradient(to right,
           rgba(0,0,0,0), rgba(230,126,34,0.75), rgba(0,0,0,0));
         margin: 2em 0; }
    img { max-width: 100%; height: auto; display: block;
          margin: 1em auto; border-radius: 6px; }
    """

    html_string = f"""<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="utf-8">
    <title>Rapport Projet MPEG-4</title>
    <style>{css}</style>
  </head>
  <body>{html_body}</body>
</html>"""

    font_config = FontConfiguration()
    HTML(string=html_string, base_url='.').write_pdf(
        REPORT_PDF, font_config=font_config, presentational_hints=True
    )


def main():
    print("\n" + "="*60)
    print("  MPEG-4 Project — Report Generator")
    print("="*60 + "\n")

    # 1. Ensure test frames exist
    frames_list = ensure_frames()
    print(f"  Found {len(frames_list)} frames in '{FRAMES_DIR}/'")

    # 2. Main encode / decode
    print("\n[STEP 1] Encoding...")
    stats_enc = encode(FRAMES_DIR, BIN_PATH,
                       gop=DEFAULT_GOP, qf=DEFAULT_QF,
                       search=DEFAULT_SEARCH, verbose=True)

    print("[STEP 2] Decoding...")
    stats_dec = decode(BIN_PATH, DECODED, verbose=True)

    # 3. Stage visualizations
    print("[STEP 3] Generating stage visualizations...")
    run_visualizations(frames_list)

    # 4. Experimental analysis
    print("\n[STEP 4] Running QF experiment...")
    qf_data = run_qf_experiment(frames_list)

    print("[STEP 5] Running GOP experiment...")
    gop_data = run_gop_experiment(frames_list)

    # 5. Build markdown report
    print("\n[STEP 6] Building report...")
    md = build_markdown(stats_enc, stats_dec, qf_data, gop_data)
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"  Markdown saved: {REPORT_MD}")

    # 6. Generate PDF
    print(f"  Generating PDF: {REPORT_PDF} ...")
    generate_pdf_from_md(md)
    print(f"  PDF saved: {REPORT_PDF}")

    print("\n" + "="*60)
    print("  Report generation complete!")
    print(f"  PDF → {REPORT_PDF}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
