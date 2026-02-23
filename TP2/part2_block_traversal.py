import cv2

# ============================================================
# PARTIE 2 : Parcours d'une image bloc par bloc
# ============================================================

# Lire l'image
img = cv2.imread('../cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    height, width, _ = img.shape

    # Définir la taille des blocs (ici : moitié de la largeur)
    block_size = int(width / 2)

    print(f"Image : {width}x{height} | Taille bloc : {block_size}px")
    print("-" * 50)

    # Parcourir l'image bloc par bloc
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Limites du bloc (pour gérer les bords)
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)

            # Extraire le bloc
            block = img[y:y_end, x:x_end]

            # Afficher les coordonnées du bloc
            print(f"Bloc ({x}, {y}) → ({x_end}, {y_end})")

            # Afficher le bloc dans une fenêtre
            cv2.imshow("Bloc", block)
            cv2.waitKey(0)    # Attendre 500ms avant le prochain bloc

    cv2.destroyAllWindows()
    print("Parcours terminé.")
