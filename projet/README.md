# MPEG-4 Simplified Video Encoder/Decoder

> Projet Multimédia — USTHB M1 — 2025/2026  
> Implémentation Python d'un pipeline de compression vidéo inspiré de MPEG-4

---

## 📦 Structure du projet

```
projet/
├── encoder.py            # Encodeur principal (CLI)
├── decoder.py            # Décodeur principal (CLI)
├── generate_frames.py    # Génère des frames de test synthétiques
├── generate_report.py    # Génère le rapport PDF complet
├── requirements.txt
├── pipeline/
│   ├── preprocessing.py  # Étape 1 : YCbCr + 4:2:0
│   ├── intra.py          # Étape 2 : I-frames (DCT + quantif + RLE)
│   ├── inter.py          # Étape 3 : P-frames (block matching + résidus)
│   ├── entropy.py        # Étape 4 : Codage entropique (zlib)
│   └── visualize.py      # Étape 5 : Visualisations matplotlib
└── utils/
    ├── metrics.py         # PSNR, SSIM, ratio de compression
    └── frames.py          # Chargement/sauvegarde des frames
```

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## 🎬 Utilisation rapide

### 1. Générer des frames de test

```bash
python3 generate_frames.py
# → crée 30 frames PNG dans frames/
```

### 2. Encoder

```bash
python3 encoder.py --input frames/ --output video.bin --gop 5 --qf 50 --search 8
```

| Option | Description | Défaut |
|---|---|---|
| `--input` | Dossier des frames sources | (requis) |
| `--output` | Fichier `.bin` de sortie | `video.bin` |
| `--gop` | Taille du groupe d'images | `10` |
| `--qf` | Facteur de qualité (1–100) | `50` |
| `--search` | Fenêtre de recherche ±S pixels | `8` |

### 3. Décoder

```bash
python3 decoder.py --input video.bin --output decoded/
```

### 4. Générer le rapport complet (PDF)

```bash
python3 generate_report.py
# → Rapport_Projet_MPEG4.pdf + visualizations/
```

---

## 🔬 Pipeline détaillé

```
Frames PNG/JPG
      │
      ▼  Stage 1 — Pré-traitement
      │  BGR → YCbCr  +  Sous-échantillonnage 4:2:0
      │
      ├─── I-frame ──► Stage 2 : DCT 8×8 → Quantif. → Zigzag → RLE
      │
      └─── P-frame ──► Stage 3 : Block Matching ±S px → Vecteurs de mouvement
                                  → Résidus → DCT → Quantif.
                       │
                       ▼  Stage 4 — Codage entropique
                          pickle + zlib (niveau 9) → video.bin
```

---

## 📊 Performances typiques (30 frames, 320×240)

| QF | Ratio | PSNR |
|---|---|---|
| 10 | ~12× | ~26 dB |
| 50 | ~4×  | ~34 dB |
| 90 | ~1.6× | ~41 dB |

---

## 📋 Format du fichier `.bin`

| Champ | Taille | Valeur |
|---|---|---|
| Magic | 4 B | `M4SP` |
| Width | 4 B | uint32 |
| Height | 4 B | uint32 |
| N_frames | 4 B | uint32 |
| GOP | 2 B | uint16 |
| QF | 2 B | uint16 |
| Data | variable | pickle + zlib-9 |
