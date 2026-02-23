import cv2
import matplotlib.pyplot as plt

# ============================================================
# PARTIE 3 : Binarisation d'une image
# ============================================================

# Lire l'image et la convertir en niveau de gris
img = cv2.imread('../cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Convertir en niveaux de gris (requis pour cv2.threshold)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Méthode 1 : Seuillage manuel (threshold = 127) ---
    threshold = 127
    _, img_bw = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

    # --- Méthode 2 : Seuillage automatique Otsu ---
    # La méthode d'Otsu minimise la variance intra-classe pour trouver
    # automatiquement le meilleur seuil basé sur l'histogramme de l'image.
    otsu_threshold, img_bw_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Seuil Otsu calculé automatiquement : {otsu_threshold}")

    # --- Affichage ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title('Image en niveaux de gris')
    axes[0].axis('off')

    axes[1].imshow(img_bw, cmap='gray')
    axes[1].set_title(f'Binarisation manuelle (seuil={threshold})')
    axes[1].axis('off')

    axes[2].imshow(img_bw_otsu, cmap='gray')
    axes[2].set_title(f'Binarisation Otsu (seuil={int(otsu_threshold)})')
    axes[2].axis('off')

    plt.suptitle("Binarisation d'une image")
    plt.tight_layout()
    plt.show()
