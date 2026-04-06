import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# ===========================================================================
# PARTIE 1 : COMPRESSION LZW
# ===========================================================================

def lzw_compress(data_str):
    """
    Compression LZW sur une chaîne de caractères.

    Paramètres :
        data_str  : chaîne de caractères à compresser

    Retourne :
        codes     : liste des codes LZW générés (entiers)
        dictionary: dictionnaire final après compression
        steps     : liste des étapes (pour affichage pédagogique)
    """
    # Initialiser le dictionnaire avec les 256 caractères ASCII
    dictionary = {chr(i): i for i in range(256)}
    dict_size = 256

    codes = []
    steps = []

    W = ""  # séquence courante (string en cours de construction)

    for C in data_str:
        WC = W + C
        if WC in dictionary:
            # WC est dans le dictionnaire → étendre la séquence
            W = WC
        else:
            # WC n'est pas dans le dictionnaire → émettre le code de W
            codes.append(dictionary[W])
            steps.append({
                'W': W,
                'C': C,
                'WC': WC,
                'code_emis': dictionary[W],
                'nouvelle_entree': f"{dict_size}: '{WC}'"
            })
            # Ajouter WC au dictionnaire
            dictionary[WC] = dict_size
            dict_size += 1
            # Recommencer avec C
            W = C

    # Émettre le code pour le dernier W
    if W:
        codes.append(dictionary[W])
        steps.append({
            'W': W,
            'C': '(fin)',
            'WC': W,
            'code_emis': dictionary[W],
            'nouvelle_entree': '(aucune)'
        })

    return codes, dictionary, steps


# ===========================================================================
# PARTIE 2 : DÉCOMPRESSION LZW (optionnel)
# ===========================================================================

def lzw_decompress(codes):
    """
    Décompression LZW : reconstitue la chaîne originale à partir des codes.

    Paramètres :
        codes     : liste des codes LZW (entiers)

    Retourne :
        result    : chaîne de caractères décompressée
    """
    # Initialiser le dictionnaire inverse (code → chaîne)
    dictionary = {i: chr(i) for i in range(256)}
    dict_size = 256

    result = []

    # Premier code
    W = dictionary[codes[0]]
    result.append(W)

    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            # Cas spécial : le code n'est pas encore dans le dico
            entry = W + W[0]
        else:
            raise ValueError(f"Code invalide : {code}")

        result.append(entry)

        # Ajouter W + entry[0] au dictionnaire
        dictionary[dict_size] = W + entry[0]
        dict_size += 1

        W = entry

    return ''.join(result)


# ===========================================================================
# PARTIE 3 : EXEMPLE PÉDAGOGIQUE — "ABABABA"
# ===========================================================================

def demo_abababa():
    print("=" * 65)
    print("  EXEMPLE PÉDAGOGIQUE : Compression LZW de 'ABABABA'")
    print("=" * 65)

    data = "ABABABA"
    codes, final_dict, steps = lzw_compress(data)

    print(f"\nDonnées en entrée : '{data}'")
    print(f"Taille originale  : {len(data)} caractères × 8 bits = {len(data) * 8} bits\n")

    print("─" * 65)
    print(f"{'Étape':<6} {'W':<8} {'C':<6} {'WC':<10} {'Code émis':<12} {'Nouvelle entrée'}")
    print("─" * 65)

    for idx, step in enumerate(steps):
        print(f"{idx+1:<6} {step['W']:<8} {step['C']:<6} {step['WC']:<10} "
              f"{step['code_emis']:<12} {step['nouvelle_entree']}")

    print("─" * 65)
    print(f"\nCodes LZW générés : {codes}")

    # Taux de compression
    # Taille originale : chaque caractère sur 8 bits
    taille_originale = len(data) * 8

    # Taille compressée : chaque code nécessite suffisamment de bits
    # Ici les codes sont ≤ 512 donc 9 bits suffisent
    bits_par_code = max(9, int(np.ceil(np.log2(max(codes) + 1))))
    taille_comprimee = len(codes) * bits_par_code

    taux = (taille_comprimee / taille_originale) * 100

    print(f"\nNombre de codes émis      : {len(codes)}")
    print(f"Bits par code             : {bits_par_code} bits")
    print(f"Taille originale          : {taille_originale} bits")
    print(f"Taille compressée         : {taille_comprimee} bits")
    print(f"Taux de compression       : {taux:.2f}%")
    print(f"Gain                      : {100 - taux:.2f}%")

    # Décompression de vérification
    decoded = lzw_decompress(codes)
    print(f"\nDécompression             : '{decoded}'")
    print(f"Vérification              : {'✓ OK' if decoded == data else '✗ ERREUR'}")

    return codes, taux


# ===========================================================================
# PARTIE 4 : COMPRESSION LZW SUR IMAGE
# ===========================================================================

def lzw_compress_image(image_path):
    print("\n" + "=" * 65)
    print(f"  COMPRESSION LZW SUR IMAGE : {os.path.basename(image_path)}")
    print("=" * 65)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image '{image_path}' introuvable.")

    height, width = img.shape
    print(f"\nImage chargée     : {image_path}")
    print(f"Dimensions        : {height} × {width} pixels")
    print(f"Pixels totaux     : {height * width}")

    # Convertir l'image en vecteur puis en chaîne de caractères
    data = np.array(img).flatten()
    data_str = ''.join([chr(p) for p in data])

    print(f"Longueur data_str : {len(data_str)} caractères")
    print(f"\nCompression LZW en cours...")

    codes, final_dict, steps = lzw_compress(data_str)

    # Calcul du taux de compression
    taille_originale = height * width * 8  # 8 bits par pixel
    bits_par_code = max(9, int(np.ceil(np.log2(len(final_dict) + 1))))
    taille_comprimee = len(codes) * bits_par_code
    taux = (taille_comprimee / taille_originale) * 100

    print(f"Codes générés     : {len(codes)}")
    print(f"Taille dictionnaire finale : {len(final_dict)} entrées")
    print(f"Bits par code     : {bits_par_code} bits")
    print(f"Taille originale  : {taille_originale} bits ({taille_originale // 8} octets)")
    print(f"Taille compressée : {taille_comprimee} bits ({taille_comprimee // 8} octets)")
    print(f"Taux de compression : {taux:.2f}%")
    print(f"Gain              : {100 - taux:.2f}%")

    return img, codes, taux


def create_test_image(output_path, size=(64, 64)):
    """Crée une image de test simple si aucune image n'est disponible."""
    img = np.zeros((size[0], size[1]), dtype=np.uint8)
    # Bandes horizontales uniformes → bonne redondance pour LZW
    for i in range(size[0]):
        val = (i // 8) * 30
        img[i, :] = min(val, 255)
    cv2.imwrite(output_path, img)
    print(f"[INIT] Image de test créée : {output_path}")
    return img


# ===========================================================================
# PARTIE 5 : VISUALISATION
# ===========================================================================

def plot_lzw_steps(steps, title="Étapes LZW"):
    """Visualise les codes émis lors de la compression."""
    codes = [s['code_emis'] for s in steps]
    labels = [s['W'] for s in steps]

    fig, ax = plt.subplots(figsize=(max(8, len(codes) * 1.2), 4))
    bars = ax.bar(range(len(codes)), codes, color='steelblue', edgecolor='white')

    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Séquence W émise", fontsize=12)
    ax.set_ylabel("Code LZW", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    for bar, code in zip(bars, codes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(code), ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig


# ===========================================================================
# PROGRAMME PRINCIPAL
# ===========================================================================

def main():
    print("=" * 65)
    print("  TP N°6 : Compression LZW")
    print("  M1 RSD / M1 IL — Multimédia — USTHB 2025/2026")
    print("=" * 65)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── ÉTAPE 1 : Exemple pédagogique ABABABA ──────────────────────────────
    codes_aba, taux_aba = demo_abababa()

    # Visualiser les étapes
    _, _, steps_aba = lzw_compress("ABABABA")
    fig1 = plot_lzw_steps(steps_aba, "Codes LZW émis pour 'ABABABA'")
    fig1_path = os.path.join(script_dir, "lzw_abababa.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Figure ABABABA sauvegardée : {fig1_path}")
    plt.show()

    # ── ÉTAPE 2 : Compression sur image ────────────────────────────────────
    # Chercher une image .bmp ou .png dans le dossier
    image_path = None
    for fname in os.listdir(script_dir):
        if fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            image_path = os.path.join(script_dir, fname)
            break

    if image_path is None:
        image_path = os.path.join(script_dir, "test_image.bmp")
        create_test_image(image_path)

    img, codes_img, taux_img = lzw_compress_image(image_path)

    # Affichage de l'image
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Compression LZW sur image", fontsize=13, fontweight='bold')

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    axes[1].hist(img.flatten(), bins=64, color='steelblue', edgecolor='none')
    axes[1].set_title("Histogramme des niveaux de gris")
    axes[1].set_xlabel("Intensité")
    axes[1].set_ylabel("Nombre de pixels")

    fig2_path = os.path.join(script_dir, "lzw_image_resultats.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Figure image sauvegardée : {fig2_path}")
    plt.show()

    # ── RÉSUMÉ FINAL ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RÉSUMÉ FINAL")
    print("=" * 65)
    print(f"  'ABABABA'   → Taux de compression : {taux_aba:.2f}%")
    print(f"  Image       → Taux de compression : {taux_img:.2f}%")
    print("=" * 65)
    print("  TP N°6 TERMINÉ AVEC SUCCÈS")
    print("=" * 65)


if __name__ == "__main__":
    main()
