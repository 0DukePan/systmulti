import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def Encode(image_path, output_file):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' introuvable.")

    height, width = image.shape
    binary_image = (image > 127).astype(np.uint8)
    fimage = binary_image.flatten()

    rle_list = []
    current_value = fimage[0]
    count = 1

    for i in range(1, len(fimage)):
        if fimage[i] == current_value:
            count += 1
        else:
            rle_list.append((count, current_value))
            current_value = fimage[i]
            count = 1
    rle_list.append((count, current_value))

    codes = []
    for c, v in rle_list:
        valeur_str = "{:03d}".format(int(v))
        count_str = str(c)
        codes.append(count_str + valeur_str)
    coded_string = ";".join(codes)

    with open(output_file, "w") as file:
        file.write(f"{height},{width}\n")
        file.write(coded_string)

    original_bits = height * width
    compressed_bits = len(coded_string) * 8
    taux = (compressed_bits / original_bits) * 100

    print(f"[ENCODE] Image : {image_path}")
    print(f"[ENCODE] Dimensions : {height} x {width}")
    print(f"[ENCODE] Pixels totaux : {height * width}")
    print(f"[ENCODE] Séquences RLE : {len(rle_list)}")
    print(f"[ENCODE] Taille chaîne codée : {len(coded_string)} caractères")
    print(f"[ENCODE] Taux de compression : {taux:.2f}%")
    print(f"[ENCODE] Fichier de sortie : {output_file}")

    return binary_image, rle_list, coded_string, taux


def Decode(input_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    dimensions = lines[0].strip().split(",")
    height = int(dimensions[0])
    width = int(dimensions[1])
    coded_string = lines[1].strip()

    print(f"\n[DECODE] Fichier : {input_file}")
    print(f"[DECODE] Dimensions attendues : {height} x {width}")
    print(f"[DECODE] Longueur chaîne codée : {len(coded_string)} caractères")

    resultat = []

    # Les codes sont séparés par ';'
    # Chaque code = compteur (variable) + valeur (3 chiffres)
    codes = coded_string.split(";")

    for code in codes:
        if len(code) < 4:
            continue
        value_str = code[-3:]   # Les 3 derniers chiffres = valeur
        count_str = code[:-3]   # Le reste = compteur
        count_val = int(count_str)
        value_val = int(value_str)
        resultat.extend([value_val] * count_val)

    decoded_image = np.array(resultat).reshape((height, width)).astype(np.uint8)

    print(f"[DECODE] Pixels décodés : {len(resultat)}")
    print(f"[DECODE] Dimensions image reconstruite : {decoded_image.shape}")

    return decoded_image


def test_codec(image_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path))

    basename = os.path.splitext(os.path.basename(image_path))[0]
    txt_file = os.path.join(output_dir, f"{basename}_rle.txt")

    print("\n" + "=" * 60)
    print(f"  TEST CODEC RLE : {os.path.basename(image_path)}")
    print("=" * 60)

    binary_image, rle_list, coded_string, taux = Encode(image_path, txt_file)

    decoded_image = Decode(txt_file)

    if np.array_equal(binary_image, decoded_image):
        print(f"\n[TEST] ✓ SUCCÈS : L'image décodée est identique à l'originale !")
    else:
        diff_count = np.sum(binary_image != decoded_image)
        print(f"\n[TEST] ✗ ÉCHEC : {diff_count} pixels différents !")

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Codec RLE — {os.path.basename(image_path)}', fontsize=14, fontweight='bold')

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Image originale\n(niveaux de gris)')
    axes[0].axis('off')

    axes[1].imshow(binary_image * 255, cmap='gray')
    axes[1].set_title('Image binarisée\n(encodée)')
    axes[1].axis('off')

    axes[2].imshow(decoded_image * 255, cmap='gray')
    axes[2].set_title(f'Image décodée\n(taux: {taux:.2f}%)')
    axes[2].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{basename}_resultats.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[TEST] Figure sauvegardée : {fig_path}")
    plt.show()

    return taux


def create_test_images(output_dir):
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img1[20:80, 20:80] = 255
    cv2.imwrite(os.path.join(output_dir, "image.bmp"), img1)
    print(f"[INIT] Image de test 'image.bmp' créée (carré blanc sur fond noir)")

    img2 = np.zeros((120, 160), dtype=np.uint8)
    for i in range(120):
        for j in range(160):
            if (i // 20 + j // 20) % 2 == 0:
                img2[i, j] = 255
            if 40 <= i <= 80 and 60 <= j <= 100:
                img2[i, j] = 200
    cv2.imwrite(os.path.join(output_dir, "cablecar.bmp"), img2)
    print(f"[INIT] Image de test 'cablecar.bmp' créée (motif en damier)")


def main():
    print("=" * 60)
    print("  TP N°5 : Compression RLE - Codec Complet (Partie 2/2)")
    print("  M1 RSD / M1 IL - Multimédia - USTHB 2025/2026")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    image_bmp = os.path.join(script_dir, "image.bmp")
    cablecar_bmp = os.path.join(script_dir, "cablecar.bmp")

    if not os.path.exists(image_bmp) or not os.path.exists(cablecar_bmp):
        print("\n[INIT] Images de test non trouvées, création automatique...")
        create_test_images(script_dir)

    taux1 = test_codec(image_bmp, script_dir)
    taux2 = test_codec(cablecar_bmp, script_dir)

    print("\n" + "=" * 60)
    print("  RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)
    print(f"  image.bmp    → Taux de compression : {taux1:.2f}%")
    print(f"  cablecar.bmp → Taux de compression : {taux2:.2f}%")
    print("=" * 60)
    print("  TP N°5 TERMINÉ AVEC SUCCÈS")
    print("=" * 60)


if __name__ == "__main__":
    main()
