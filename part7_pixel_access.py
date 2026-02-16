import cv2
import numpy as np

# Read the image
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    print(f"Original shape: {img.shape}")
    height, width, channels = img.shape
    
    # EXEMPLE 1: Accéder à une ligne spécifique
    # On prend la ligne du milieu
    mid_row_idx = height // 2
    row_data = img[mid_row_idx, :, :]
    print(f"Données de la ligne {mid_row_idx} (shape): {row_data.shape}")
    
    # EXEMPLE 2: Accéder à une colonne spécifique
    # On prend la colonne du milieu
    mid_col_idx = width // 2
    col_data = img[:, mid_col_idx, :]
    print(f"Données de la colonne {mid_col_idx} (shape): {col_data.shape}")
    
    # EXEMPLE 3: Modification de pixels
    # Créer une copie pour ne pas modifier l'original tout de suite
    img_mod = img.copy()
    
    # Tracer une ligne horizontale blanche au milieu
    img_mod[mid_row_idx, :, :] = [255, 255, 255] # BGR pour Blanc
    
    # Tracer une ligne verticale noire au milieu
    img_mod[:, mid_col_idx, :] = [0, 0, 0] # BGR pour Noir
    
    # Modifier une zone rectangulaire (100x100 pixels) en haut à gauche
    img_mod[0:100, 0:100] = [0, 255, 0] # Vert
    
    cv2.imshow("Acces Pixels et Modifications", img_mod)
    print("Appuyez sur une touche pour fermer la fenêtre...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
