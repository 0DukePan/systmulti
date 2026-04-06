import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

def load_image_grayscale(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Erreur : Le fichier image '{image_path}' n'existe pas.\nVeuillez vérifier le chemin et réessayer.")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Erreur : Impossible de lire l'image '{image_path}'.\nLe fichier existe mais n'est pas un format image valide.")
    print(f'[INFO] Image chargée avec succès : {image_path}')
    print(f"[INFO] Dimensions de l'image : {image.shape[0]} x {image.shape[1]} pixels")
    print(f'[INFO] Type de données : {image.dtype}')
    print(f'[INFO] Valeur min : {image.min()}, Valeur max : {image.max()}')
    return image

def binarize_image(gray_image, threshold=127):
    binary_image = (gray_image > threshold).astype(np.uint8)
    total_pixels = binary_image.size
    white_pixels = np.sum(binary_image == 1)
    black_pixels = np.sum(binary_image == 0)
    print(f"\n[INFO] === Binarisation de l'image ===")
    print(f'[INFO] Seuil de binarisation : {threshold}')
    print(f'[INFO] Nombre total de pixels : {total_pixels}')
    print(f'[INFO] Pixels blancs (1) : {white_pixels} ({100 * white_pixels / total_pixels:.2f}%)')
    print(f'[INFO] Pixels noirs  (0) : {black_pixels} ({100 * black_pixels / total_pixels:.2f}%)')
    return binary_image

def flatten_image(binary_image):
    fimage = binary_image.flatten()
    print(f"\n[INFO] === Aplatissement de l'image ===")
    print(f'[INFO] Dimensions originales : {binary_image.shape}')
    print(f'[INFO] Dimension après aplatissement : {fimage.shape}')
    print(f'[INFO] Nombre total de pixels : {len(fimage)}')
    preview_length = min(50, len(fimage))
    print(f'[INFO] Premiers {preview_length} pixels : {list(fimage[:preview_length])}')
    return fimage

def rle_encode(flat_array):
    if len(flat_array) == 0:
        print('[WARN] Le tableau est vide, pas de données à encoder.')
        return []
    rle_list = []
    current_value = flat_array[0]
    count = 1
    print(f'\n[INFO] === Encodage RLE en cours ===')
    print(f"[INFO] Taille du tableau d'entrée : {len(flat_array)} pixels")
    for i in range(1, len(flat_array)):
        if flat_array[i] == current_value:
            count += 1
        else:
            rle_list.append((count, current_value))
            current_value = flat_array[i]
            count = 1
    rle_list.append((count, current_value))
    print(f'[INFO] Nombre de séquences RLE : {len(rle_list)}')
    print(f'[INFO] Premières 10 séquences : {rle_list[:10]}')
    return rle_list

def rle_to_binary_string(rle_list, bits_per_count=3, bits_per_value=3):
    coded_string = ''
    for (count, value) in rle_list:
        count_str = '{:0{width}d}'.format(count, width=bits_per_count)
        value_str = '{:0{width}d}'.format(int(value), width=bits_per_value)
        coded_string += count_str + value_str
    return coded_string

def rle_to_string_format(rle_list):
    coded_string = ''
    for (count, value) in rle_list:
        valeur_str = '{:03d}'.format(int(value))
        count_str = str(count)
        coded_string += count_str + valeur_str
    print(f'\n[INFO] === Chaîne RLE générée ===')
    print(f'[INFO] Longueur de la chaîne : {len(coded_string)} caractères')
    preview = coded_string[:80] if len(coded_string) > 80 else coded_string
    print(f'[INFO] Aperçu : {preview}...')
    return coded_string

def write_rle_to_file(coded_string, file_name):
    with open(file_name, 'w') as file:
        file.write(coded_string)
    print(f'\n[INFO] === Écriture du fichier compressé ===')
    print(f'[INFO] Fichier : {file_name}')
    print(f'[INFO] Taille du fichier : {os.path.getsize(file_name)} octets')

def read_rle_from_file(file_name):
    with open(file_name, 'r') as file:
        txt = file.readlines()[0]
    print(f'\n[INFO] === Lecture du fichier compressé ===')
    print(f'[INFO] Fichier : {file_name}')
    print(f'[INFO] Longueur de la chaîne lue : {len(txt)} caractères')
    return txt

def rle_decode(rle_list):
    decoded = []
    for (count, value) in rle_list:
        decoded.extend([int(value)] * count)
    decoded_array = np.array(decoded, dtype=np.uint8)
    print(f'\n[INFO] === Décodage RLE ===')
    print(f'[INFO] Nombre de séquences : {len(rle_list)}')
    print(f'[INFO] Taille du tableau décodé : {len(decoded_array)} pixels')
    return decoded_array

def reconstruct_image(decoded_array, original_shape):
    expected_size = original_shape[0] * original_shape[1]
    if len(decoded_array) != expected_size:
        raise ValueError(f'Erreur : Le tableau décodé ({len(decoded_array)} pixels) ne correspond pas aux dimensions attendues ({original_shape[0]}x{original_shape[1]} = {expected_size} pixels).')
    reconstructed = decoded_array.reshape(original_shape)
    print(f"\n[INFO] === Reconstruction de l'image ===")
    print(f'[INFO] Dimensions reconstruites : {reconstructed.shape}')
    return reconstructed

def calculate_compression_ratio(original_size_bits, compressed_size_bits):
    if original_size_bits == 0:
        return 0.0
    taux = compressed_size_bits / original_size_bits * 100
    print(f'\n[INFO] === Taux de compression ===')
    print(f'[INFO] Taille originale    : {original_size_bits} bits')
    print(f'[INFO] Taille compressée   : {compressed_size_bits} bits')
    print(f'[INFO] Taux de compression : {taux:.2f}%')
    print(f'[INFO] Gain de compression : {100 - taux:.2f}%')
    if taux < 50:
        print(f'[INFO] ✓ Excellente compression (< 50%)')
    elif taux < 75:
        print(f'[INFO] ✓ Bonne compression (< 75%)')
    elif taux < 100:
        print(f'[INFO] △ Compression modérée (< 100%)')
    else:
        print(f'[INFO] ✗ Pas de gain de compression (≥ 100%)')
    return taux

def calculate_compression_stats(binary_image, rle_list, coded_string):
    stats = {}
    (hauteur, largeur) = binary_image.shape
    stats['hauteur'] = hauteur
    stats['largeur'] = largeur
    stats['total_pixels'] = hauteur * largeur
    stats['taille_originale_bits'] = stats['total_pixels']
    stats['taille_code'] = len(coded_string)
    stats['taille_code_bits'] = len(coded_string) * 8
    stats['nombre_sequences'] = len(rle_list)
    if rle_list:
        counts = [c for (c, v) in rle_list]
        stats['longueur_moyenne_sequence'] = np.mean(counts)
        stats['longueur_max_sequence'] = max(counts)
        stats['longueur_min_sequence'] = min(counts)
    else:
        stats['longueur_moyenne_sequence'] = 0
        stats['longueur_max_sequence'] = 0
        stats['longueur_min_sequence'] = 0
    stats['taux_compression'] = calculate_compression_ratio(stats['taille_originale_bits'], stats['taille_code_bits'])
    stats['ratio_noir'] = np.sum(binary_image == 0) / stats['total_pixels']
    stats['ratio_blanc'] = np.sum(binary_image == 1) / stats['total_pixels']
    return stats

def display_results(original_gray, binary_image, reconstructed_image, stats):
    (fig, axes) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TP4 : Compression RLE - Résultats', fontsize=16, fontweight='bold')
    axes[0, 0].imshow(original_gray, cmap='gray')
    axes[0, 0].set_title('Image originale (niveaux de gris)', fontsize=12)
    axes[0, 0].axis('off')
    axes[0, 1].imshow(binary_image, cmap='gray')
    axes[0, 1].set_title('Image binarisée (seuil = 127)', fontsize=12)
    axes[0, 1].axis('off')
    axes[1, 0].imshow(reconstructed_image, cmap='gray')
    axes[1, 0].set_title('Image reconstruite (après décompression)', fontsize=12)
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    stats_text = f"=== Statistiques de Compression RLE ===\n\nDimensions image : {stats['hauteur']} × {stats['largeur']}\nNombre total de pixels : {stats['total_pixels']}\n\nTaille originale : {stats['taille_originale_bits']} bits\nTaille compressée : {stats['taille_code_bits']} bits\nTaux de compression : {stats['taux_compression']:.2f}%\nGain : {100 - stats['taux_compression']:.2f}%\n\nNombre de séquences RLE : {stats['nombre_sequences']}\nLongueur moy. des séquences : {stats['longueur_moyenne_sequence']:.1f}\nPlus longue séquence : {stats['longueur_max_sequence']}\n\nRatio pixels noirs : {stats['ratio_noir'] * 100:.1f}%\nRatio pixels blancs : {stats['ratio_blanc'] * 100:.1f}%"
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Statistiques de compression', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'resultats_rle.png'), dpi=150, bbox_inches='tight')
    print(f'\n[INFO] Figure sauvegardée : resultats_rle.png')
    plt.show()

def display_rle_visualization(rle_list, max_sequences=50):
    display_list = rle_list[:max_sequences]
    counts = [c for (c, v) in display_list]
    values = [int(v) for (c, v) in display_list]
    indices = range(len(display_list))
    colors = ['black' if v == 0 else 'gold' for v in values]
    (fig, ax) = plt.subplots(figsize=(16, 5))
    bars = ax.bar(indices, counts, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_xlabel('Numéro de séquence', fontsize=12)
    ax.set_ylabel('Nombre de répétitions', fontsize=12)
    ax.set_title(f'Visualisation des séquences RLE (premières {max_sequences})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='black', edgecolor='gray', label='Pixel noir (0)'), Patch(facecolor='gold', edgecolor='gray', label='Pixel blanc (1)')]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'visualisation_rle.png'), dpi=150, bbox_inches='tight')
    print(f'[INFO] Figure sauvegardée : visualisation_rle.png')
    plt.show()

def create_test_image(size=(12, 24)):
    (hauteur, largeur) = size
    test_image = np.zeros((hauteur, largeur), dtype=np.uint8)
    for i in range(hauteur):
        if i % 3 == 0:
            test_image[i, :] = 1
        elif i % 3 == 1:
            test_image[i, :largeur // 2] = 0
            test_image[i, largeur // 2:] = 1
        else:
            for j in range(largeur):
                test_image[i, j] = j % 2
    print(f'\n[INFO] === Image de test créée ===')
    print(f'[INFO] Dimensions : {test_image.shape}')
    print(f'[INFO] Pixels blancs : {np.sum(test_image == 1)}')
    print(f'[INFO] Pixels noirs  : {np.sum(test_image == 0)}')
    return test_image

def create_example_from_tp():
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION AVEC L'EXEMPLE DU TP")
    print(f"{'=' * 60}")
    expected_code = '002001351131013411310003110131113201331'
    expected_ratio = 29.17
    print(f'[INFO] Code attendu : {expected_code}')
    print(f'[INFO] Taille du code attendue : 84 bits')
    print(f'[INFO] Taux de compression attendu : {expected_ratio}%')
    return (expected_code, expected_ratio)

def main():
    print('=' * 70)
    print('  TP N°4 : Compression de données avec RLE')
    print('  M1 RSD / M1 IL - Multimédia - USTHB 2025/2026')
    print('=' * 70)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'compressed_rle.txt')
    create_example_from_tp()
    print(f"\n{'=' * 60}")
    print(f"  ÉTAPE 1 : CHARGEMENT / CRÉATION DE L'IMAGE")
    print(f"{'=' * 60}")
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_found = None
    for f in os.listdir(script_dir):
        if any((f.lower().endswith(ext) for ext in image_extensions)):
            image_found = os.path.join(script_dir, f)
            break
    if image_found:
        print(f'[INFO] Image trouvée : {image_found}')
        gray_image = load_image_grayscale(image_found)
    else:
        print("[INFO] Aucune image trouvée, création d'une image de test...")
        gray_image = create_test_image(size=(12, 24))
        test_path = os.path.join(script_dir, 'test_image.png')
        cv2.imwrite(test_path, gray_image * 255)
        print(f'[INFO] Image de test sauvegardée : {test_path}')
    print(f"\n{'=' * 60}")
    print(f"  ÉTAPE 2 : BINARISATION DE L'IMAGE")
    print(f"{'=' * 60}")
    binary_image = binarize_image(gray_image, threshold=127)
    print(f"\n{'=' * 60}")
    print(f"  ÉTAPE 3 : APLATISSEMENT DE L'IMAGE (FLATTEN)")
    print(f"{'=' * 60}")
    fimage = flatten_image(binary_image)
    print(f"\n{'=' * 60}")
    print(f'  ÉTAPE 4 : ENCODAGE RLE')
    print(f"{'=' * 60}")
    rle_list = rle_encode(fimage)
    coded_string = rle_to_string_format(rle_list)
    print(f"\n{'=' * 60}")
    print(f'  ÉTAPE 5 : ÉCRITURE DU FICHIER COMPRESSÉ')
    print(f"{'=' * 60}")
    write_rle_to_file(coded_string, output_file)
    read_back = read_rle_from_file(output_file)
    assert read_back == coded_string, 'Erreur : Le fichier lu ne correspond pas au code écrit !'
    print('[INFO] ✓ Vérification de lecture/écriture réussie')
    print(f"\n{'=' * 60}")
    print(f'  ÉTAPE 6 : CALCUL DU TAUX DE COMPRESSION')
    print(f"{'=' * 60}")
    stats = calculate_compression_stats(binary_image, rle_list, coded_string)
    print(f"\n{'=' * 60}")
    print(f'  RÉSUMÉ DE LA COMPRESSION')
    print(f"{'=' * 60}")
    print(f"  Image              : {stats['hauteur']}×{stats['largeur']} pixels")
    print(f"  Nombre de pixels   : {stats['total_pixels']}")
    print(f"  Taille originale   : {stats['taille_originale_bits']} bits")
    print(f"  Taille compressée  : {stats['taille_code_bits']} bits")
    print(f"  Taux de compression: {stats['taux_compression']:.2f}%")
    print(f"  Gain               : {100 - stats['taux_compression']:.2f}%")
    print(f"  Séquences RLE      : {stats['nombre_sequences']}")
    print(f"{'=' * 60}")
    print(f"\n{'=' * 60}")
    print(f'  ÉTAPE 7 : DÉCOMPRESSION ET VÉRIFICATION')
    print(f"{'=' * 60}")
    decoded_array = rle_decode(rle_list)
    reconstructed_image = reconstruct_image(decoded_array, binary_image.shape)
    if np.array_equal(binary_image, reconstructed_image):
        print("[INFO] ✓ SUCCÈS : L'image reconstruite est identique à l'originale !")
        print('[INFO]   La compression RLE est bien sans perte (lossless).')
    else:
        print("[ERREUR] ✗ ÉCHEC : L'image reconstruite diffère de l'originale !")
        diff_count = np.sum(binary_image != reconstructed_image)
        print(f'[ERREUR]   Nombre de pixels différents : {diff_count}')
    print(f"\n{'=' * 60}")
    print(f'  ÉTAPE 8 : AFFICHAGE DES RÉSULTATS')
    print(f"{'=' * 60}")
    try:
        display_results(gray_image, binary_image, reconstructed_image, stats)
        display_rle_visualization(rle_list)
    except Exception as e:
        print(f"[WARN] Impossible d'afficher les graphiques : {e}")
        print('[WARN] Les résultats sont disponibles dans la console.')
    print(f"\n{'=' * 70}")
    print(f'  TP N°4 TERMINÉ AVEC SUCCÈS')
    print(f'  Fichier compressé : {output_file}')
    print(f"{'=' * 70}")
if __name__ == '__main__':
    main()