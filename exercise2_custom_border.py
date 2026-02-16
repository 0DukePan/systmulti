import cv2
import numpy as np
import matplotlib.pyplot as plt

def solve_border_exercise():
    # 1. Charger l'image
    img = cv2.imread('cablecar.bmp')
    if img is None:
        print("Erreur: Image non trouvee")
        return

    # Convertir en RGB pour Matplotlib
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dimensions des bordures (en pixels)
    pad_t, pad_b, pad_l, pad_r = 20, 20, 20, 20

    # 2. Créer la bordure colorée
    # Base noire
    padded_img = np.pad(rgb_img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant', constant_values=0)

    # Ajout des couleurs (Top=Vert, Bottom=Rouge, Left=Bleu)
    padded_img[0:pad_t, :, 1] = 255  # Vert
    padded_img[-pad_b:, :, 0] = 255  # Rouge
    padded_img[:, 0:pad_l, 2] = 255  # Bleu
    # Droite reste Noir (0)

    # 3. Affichage Side-by-Side comme sur la figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Image Originale
    axes[0].imshow(rgb_img)
    axes[0].set_title("l’image originale")
    axes[0].axis('off')

    # Image avec Padding
    axes[1].imshow(padded_img)
    axes[1].set_title("apres le padding")
    axes[1].axis('off')

    plt.show()

    # Sauvegarde
    bgr_final = cv2.cvtColor(padded_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('exercice_border_final.png', bgr_final)
    print("Image sauvegardée.")

if __name__ == "__main__":
    solve_border_exercise()
