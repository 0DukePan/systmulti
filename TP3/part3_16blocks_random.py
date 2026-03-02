import cv2
import numpy as np
import random

# ==========================================
# PART 3 - 16-Block Random Filter Exercise
# ==========================================

# Read grayscale image
gray_img = cv2.imread('../cablecar.bmp', cv2.IMREAD_GRAYSCALE)

if gray_img is None:
    print("Error: 'cablecar.bmp' not found in the parent directory.")
else:
    # Define the set of available kernels
    k_laplacian = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
    k_emboss    = np.array([[-2, -1,  0], [-1,  1,  1], [ 0,  1,  2]])
    k_sharpen   = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]])
    k_sobel_v   = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])
    k_gaussian  = np.array([[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]]) / 16.0

    kernels_list = [k_laplacian, k_emboss, k_sharpen, k_sobel_v, k_gaussian]

    height, width = gray_img.shape

    # Diviser l'image en 16 blocs de même dimension (4x4 grid)
    h_b, w_b = height // 4, width // 4
    final_img = np.zeros_like(gray_img)

    for i in range(4):
        for j in range(4):
            # Block boundaries
            y1, y2 = i * h_b, (i + 1) * h_b
            x1, x2 = j * w_b, (j + 1) * w_b

            block = gray_img[y1:y2, x1:x2]

            # Choisir un noyau aléatoirement et l'appliquer au bloc
            chosen_k = random.choice(kernels_list)
            # ddepth=-1 keeps output the same depth as source (uint8)
            final_img[y1:y2, x1:x2] = cv2.filter2D(src=block, ddepth=-1, kernel=chosen_k)

    # Regrouper les 16 blocs filtrés dans une seule image et l'afficher
    cv2.imshow('Original Grayscale', gray_img)
    cv2.imshow('Part 3 - 16 Blocks Random Filters', final_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
