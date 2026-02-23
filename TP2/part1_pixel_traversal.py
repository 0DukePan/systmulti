import cv2
import matplotlib.pyplot as plt

# ============================================================
# PARTIE 1 : Parcours d'une image pixel par pixel
# ============================================================

# Lire l'image
img = cv2.imread('../cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    height, width, _ = img.shape
    print(f"Dimensions : {width}x{height}")

    # --- Méthode 1 : Parcours par boucle ---
    img_loop = img.copy()
    for y in range(height):
        for x in range(width):
            pixel = img_loop[y, x]
            B, G, R = pixel
            # Modifier une région spécifique : x in (100,200), y in (100,150)
            if 100 < x < 200 and 100 < y < 150:
                img_loop[y, x] = (255, 0, 0)   # BGR Bleu

    # --- Méthode 2 : Accès numpy direct (beaucoup plus rapide) ---
    img_numpy = img.copy()
    img_numpy[100:150, 100:200] = (255, 0, 0)  # Même résultat en une ligne

    # Affichage comparatif
    rgb_loop  = cv2.cvtColor(img_loop,  cv2.COLOR_BGR2RGB)
    rgb_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(rgb_loop)
    axes[0].set_title("Méthode Boucle (lente)")
    axes[0].axis('off')

    axes[1].imshow(rgb_numpy)
    axes[1].set_title("Méthode NumPy (rapide)")
    axes[1].axis('off')

    plt.suptitle("Parcours & Modification de pixels")
    plt.tight_layout()
    plt.show()
