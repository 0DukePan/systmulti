import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# PARTIE 1 : EXEMPLE PÉDAGOGIQUE — Bloc 2×2 dans zone 4×4
# ===========================================================================

def compute_mse(block1, block2):
    """Calcule le MSE entre deux blocs de même taille."""
    n = block1.shape[0] * block1.shape[1]
    return np.sum((block1.astype(float) - block2.astype(float)) ** 2) / n


def demo_2x2():
    print("=" * 65)
    print("  PARTIE 1 : Exemple pédagogique — Bloc 2×2 / Zone 4×4")
    print("=" * 65)

    # Bloc courant
    Bcourant = np.array([[10, 12],
                         [14, 16]], dtype=float)

    # Zone de recherche 4×4
    img = np.array([[10, 11, 12, 13],
                    [14, 15, 16, 17],
                    [18, 19, 20, 21],
                    [22, 23, 24, 25]], dtype=float)

    print(f"\nBloc courant :\n{Bcourant.astype(int)}")
    print(f"\nZone de recherche 4×4 :\n{img.astype(int)}")

    block_size = 2
    zone_h, zone_w = img.shape
    results = []

    print(f"\n{'─'*55}")
    print(f"{'(dy,dx)':<12} {'Bloc candidat':<25} {'MSE':>8}")
    print(f"{'─'*55}")

    for dy in range(zone_h - block_size + 1):
        for dx in range(zone_w - block_size + 1):
            candidate = img[dy:dy + block_size, dx:dx + block_size]
            mse = compute_mse(Bcourant, candidate)
            results.append((mse, dy, dx, candidate.copy()))
            bloc_str = str(candidate.astype(int).tolist())
            print(f"({dy},{dx}){'':<7} {bloc_str:<25} {mse:>8.2f}")

    print(f"{'─'*55}")

    # Meilleur bloc = MSE minimale
    best = min(results, key=lambda r: r[0])
    best_mse, best_dy, best_dx, best_bloc = best

    print(f"\n✓ Meilleur bloc trouvé à (dy={best_dy}, dx={best_dx})")
    print(f"  Vecteur de mouvement : (dx={best_dx}, dy={best_dy})")
    print(f"  MSE minimale         : {best_mse:.4f}")
    print(f"  Bloc candidat        :\n{best_bloc.astype(int)}")

    # Résidu = Bcourant - meilleur_bloc
    residu = Bcourant - best_bloc
    print(f"\n  Résidu (Bcourant - meilleur bloc) :\n{residu.astype(int)}")

    return best_dx, best_dy, residu


# ===========================================================================
# PARTIE 2 : APPLICATION SUR IMAGES (frame_1 et frame_2)
# ===========================================================================

def find_best_block(img_prev, block_curr, bx, by, block_size=16, search_range=4):
    """
    Cherche le meilleur bloc dans img_prev correspondant à block_curr.

    Paramètres :
        img_prev     : image précédente (frame_1)
        block_curr   : bloc courant extrait de img_2
        bx, by       : position du bloc dans img_2
        block_size   : taille du bloc (16×16)
        search_range : rayon de recherche (±4 pixels)

    Retourne :
        best_dx, best_dy : vecteur de mouvement
        best_mse         : MSE minimale obtenue
        best_block       : meilleur bloc trouvé dans img_prev
    """
    h, w = img_prev.shape
    best_mse = float('inf')
    best_dx, best_dy = 0, 0
    best_block = None

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ry = by + dy
            rx = bx + dx
            # Vérifier que le bloc candidat est dans les limites de l'image
            if ry < 0 or rx < 0 or ry + block_size > h or rx + block_size > w:
                continue
            candidate = img_prev[ry:ry + block_size, rx:rx + block_size]
            mse = compute_mse(block_curr, candidate)
            if mse < best_mse:
                best_mse = mse
                best_dx, best_dy = dx, dy
                best_block = candidate.copy()

    return best_dx, best_dy, best_mse, best_block


def motion_estimation(img1_path, img2_path, block_size=16, search_range=4):
    """
    Calcule les vecteurs de mouvement et les résidus entre deux trames.

    Paramètres :
        img1_path    : chemin de frame_1 (image précédente)
        img2_path    : chemin de frame_2 (image courante)
        block_size   : taille des blocs (16×16)
        search_range : rayon de recherche (±4 pixels)

    Retourne :
        motion_vectors : dictionnaire {(bx,by): (dx,dy)}
        residuals      : dictionnaire {(bx,by): résidu 2D}
        img1, img2     : images chargées
    """
    print("\n" + "=" * 65)
    print("  PARTIE 2 : Estimation de mouvement sur images réelles")
    print("=" * 65)

    # Charger les images en niveaux de gris
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"Image '{img1_path}' introuvable.")
    if img2 is None:
        raise FileNotFoundError(f"Image '{img2_path}' introuvable.")

    # Redimensionner si nécessaire pour que les dimensions soient multiples de block_size
    h, w = img2.shape
    h2 = (h // block_size) * block_size
    w2 = (w // block_size) * block_size
    img1 = img1[:h2, :w2]
    img2 = img2[:h2, :w2]

    print(f"\nframe_1 : {img1_path} → {img1.shape}")
    print(f"frame_2 : {img2_path} → {img2.shape}")
    print(f"Taille de bloc : {block_size}×{block_size}")
    print(f"Zone de recherche : ±{search_range} pixels")
    print(f"Blocs à traiter : {(h2 // block_size) * (w2 // block_size)}")

    motion_vectors = {}
    residuals = {}
    total_mse = 0
    num_blocks = 0

    nb_y = h2 // block_size
    nb_x = w2 // block_size

    for ib in range(nb_y):
        for jb in range(nb_x):
            by = ib * block_size
            bx = jb * block_size

            # Extraire le bloc courant de img_2
            block_curr = img2[by:by + block_size, bx:bx + block_size].astype(float)

            # Chercher le meilleur bloc dans img_1
            dx, dy, mse, best_block = find_best_block(
                img1.astype(float), block_curr, bx, by, block_size, search_range
            )

            # Résidu
            if best_block is not None:
                residual = block_curr - best_block
            else:
                residual = block_curr

            motion_vectors[(bx, by)] = (dx, dy)
            residuals[(bx, by)] = residual
            total_mse += mse
            num_blocks += 1

    avg_mse = total_mse / num_blocks if num_blocks > 0 else 0
    print(f"\nBlocs traités       : {num_blocks}")
    print(f"MSE moyen           : {avg_mse:.4f}")
    print(f"Vecteurs non nuls   : {sum(1 for v in motion_vectors.values() if v != (0, 0))}")

    # Afficher les résidus
    print("\n── Résidus (min/max par bloc) ──")
    for (bx, by), res in list(residuals.items())[:10]:
        print(f"  Bloc ({by:4d},{bx:4d}) → résidu min={res.min():.1f}, max={res.max():.1f}, mean={res.mean():.2f}")
    if len(residuals) > 10:
        print(f"  ... ({len(residuals) - 10} autres blocs)")

    return motion_vectors, residuals, img1, img2


def visualize_motion_vectors(img2, motion_vectors, block_size=16, output_path=None):
    """
    Visualise les vecteurs de mouvement superposés sur img_2.
    Utilise plt.quiver() comme demandé dans l'énoncé.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img2, cmap='gray')
    ax.set_title("Vecteurs de mouvement (estimation par blocs 16×16, zone ±4)",
                 fontsize=13, fontweight='bold')

    # Préparer les données pour quiver
    X, Y, DX, DY = [], [], [], []

    for (bx, by), (dx, dy) in motion_vectors.items():
        cx = bx + block_size // 2  # centre du bloc
        cy = by + block_size // 2
        X.append(cx)
        Y.append(cy)
        DX.append(dx)
        DY.append(-dy)  # inverser y pour correspondre à l'axe image

    if X:
        ax.quiver(X, Y, DX, DY,
                  color='lime', scale=50, scale_units='xy',
                  angles='xy', width=0.003, headwidth=4, headlength=5,
                  alpha=0.85)

    # Grille des blocs
    h, w = img2.shape
    for y in range(0, h, block_size):
        ax.axhline(y, color='white', linewidth=0.3, alpha=0.4)
    for x in range(0, w, block_size):
        ax.axvline(x, color='white', linewidth=0.3, alpha=0.4)

    ax.set_xlabel("Colonnes (pixels)", fontsize=11)
    ax.set_ylabel("Lignes (pixels)", fontsize=11)

    legend = [mpatches.Patch(color='lime', label='Vecteur de mouvement (dx, dy)')]
    ax.legend(handles=legend, loc='upper right', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"[INFO] Figure vecteurs de mouvement : {output_path}")
    plt.show()


def visualize_residuals(residuals, img2, block_size=16, output_path=None):
    """
    Affiche une carte des résidus reconstruite depuis tous les blocs.
    """
    h, w = img2.shape
    residual_map = np.zeros((h, w), dtype=float)

    for (bx, by), res in residuals.items():
        rh = min(block_size, h - by)
        rw = min(block_size, w - bx)
        residual_map[by:by + rh, bx:bx + rw] = np.abs(res[:rh, :rw])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Estimation de mouvement — Résultats", fontsize=14, fontweight='bold')

    axes[0].imshow(img2, cmap='gray')
    axes[0].set_title("frame_2 (image courante)")
    axes[0].axis('off')

    im = axes[1].imshow(residual_map, cmap='hot', vmin=0, vmax=50)
    axes[1].set_title("Carte des résidus |Bcourant - Bpréc|")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    hist_vals = residual_map.flatten()
    axes[2].hist(hist_vals, bins=64, color='steelblue', edgecolor='none')
    axes[2].set_title("Distribution des résidus")
    axes[2].set_xlabel("Valeur du résidu")
    axes[2].set_ylabel("Nombre de pixels")
    axes[2].axvline(hist_vals.mean(), color='red', linestyle='--',
                    label=f"Moyenne = {hist_vals.mean():.1f}")
    axes[2].legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"[INFO] Figure résidus : {output_path}")
    plt.show()


# ===========================================================================
# PROGRAMME PRINCIPAL
# ===========================================================================

def main():
    print("=" * 65)
    print("  TP N°8 : Compression Vidéo — Compensation de Mouvement")
    print("  M1 RSD / M1 IL — Multimédia — USTHB 2025/2026")
    print("=" * 65)

    # ── ÉTAPE 1 : Exemple pédagogique 2×2 ─────────────────────────────────
    best_dx, best_dy, residu = demo_2x2()

    # ── ÉTAPE 2 : Estimation de mouvement sur les images réelles ───────────
    img1_path = os.path.join(SCRIPT_DIR, "frame_1.png")
    img2_path = os.path.join(SCRIPT_DIR, "frame_2.png")

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("\n[ERREUR] frame_1.png ou frame_2.png introuvable dans le répertoire.")
        return

    motion_vectors, residuals, img1, img2 = motion_estimation(
        img1_path, img2_path, block_size=16, search_range=4
    )

    # ── ÉTAPE 3 : Visualisations ───────────────────────────────────────────
    print("\n[INFO] Génération des visualisations...")

    quiver_path = os.path.join(SCRIPT_DIR, "motion_vectors.png")
    residuals_path = os.path.join(SCRIPT_DIR, "residuals.png")

    visualize_motion_vectors(img2, motion_vectors, block_size=16, output_path=quiver_path)
    visualize_residuals(residuals, img2, block_size=16, output_path=residuals_path)

    # ── RÉSUMÉ ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RÉSUMÉ")
    print("=" * 65)
    print(f"  Exemple 2×2  → Vecteur : (dx={best_dx}, dy={best_dy}), Résidu = {residu.astype(int).tolist()}")
    print(f"  Images réelles → {len(motion_vectors)} blocs 16×16 traités")
    print(f"  Figures : {quiver_path}")
    print(f"            {residuals_path}")
    print("=" * 65)
    print("  TP N°8 TERMINÉ AVEC SUCCÈS")
    print("=" * 65)


if __name__ == "__main__":
    main()
