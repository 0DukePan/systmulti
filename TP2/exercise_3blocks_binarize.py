import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXERCICE : Diviser l'image en 3 blocs VERTICAUX
#            et binariser UNIQUEMENT le bloc du milieu
# ============================================================

# Lire l'image
img = cv2.imread('../cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    height, width, _ = img.shape

    # --- Calcul des limites des 3 blocs verticaux ---
    # Les blocs sont séparés horizontalement (selon la largeur / axe X)
    # Bloc Gauche : colonnes [0 : width/3]
    # Bloc Milieu : colonnes [width/3 : 2*width/3]
    # Bloc Droite : colonnes [2*width/3 : width]
    w1 = width  // 3          # fin du bloc gauche = début du bloc milieu
    w2 = 2 * width // 3       # fin du bloc milieu = début du bloc droite

    # --- Extraire les 3 blocs ---
    bloc_gauche = img[:, 0:w1]          # toutes les lignes, colonnes 0 → w1
    bloc_milieu = img[:, w1:w2]         # toutes les lignes, colonnes w1 → w2
    bloc_droite = img[:, w2:width]      # toutes les lignes, colonnes w2 → fin

    # --- Binariser le bloc du milieu ---
    # Convertir en niveaux de gris d'abord
    milieu_gray = cv2.cvtColor(bloc_milieu, cv2.COLOR_BGR2GRAY)
    # Seuillage automatique Otsu
    otsu_thresh, milieu_binarise = cv2.threshold(milieu_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Seuil Otsu pour le bloc milieu : {int(otsu_thresh)}")

    # --- Reconstruire l'image finale ---
    # Bloc gauche et droite restent en couleur (BGR→RGB pour Matplotlib)
    gauche_rgb = cv2.cvtColor(bloc_gauche, cv2.COLOR_BGR2RGB)
    droite_rgb = cv2.cvtColor(bloc_droite, cv2.COLOR_BGR2RGB)

    # Le bloc milieu binarisé est en niveaux de gris (1 canal),
    # on le convertit en RGB (3 canaux) pour pouvoir concaténer
    milieu_rgb = cv2.cvtColor(milieu_binarise, cv2.COLOR_GRAY2RGB)

    # Concatenation horizontale des 3 blocs
    result = np.concatenate([gauche_rgb, milieu_rgb, droite_rgb], axis=1)

    #  Affichage Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Image originale
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    # Résultat avec bloc milieu binarisé
    axes[1].imshow(result)
    axes[1].set_title("Bloc milieu binarisé (Otsu)")
    axes[1].axis('off')

    # Dessiner les séparations verticales sur l'image résultat
    axes[1].axvline(x=w1, color='red', linewidth=2, linestyle='--', label='Séparations')
    axes[1].axvline(x=w2, color='red', linewidth=2, linestyle='--')
    axes[1].legend(loc='upper right')

    plt.suptitle("Exercice TP2 : 3 Blocs Verticaux – Binarisation du Bloc Milieu")
    plt.tight_layout()
    plt.show()
