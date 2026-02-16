import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Separate RGB images (Note: OpenCV reads as BGR)
    B, G, R = cv2.split(img)
    
    # Merge back to RGB (OpenCV merge order)
    RGB_img = cv2.merge([R, G, B])
    
    plt.figure()
    plt.imshow(RGB_img)
    plt.axis('off')
    plt.title('Merged RGB Image')
    plt.show()

    # Create empty images of the same shape
    R_img = np.zeros_like(img)
    G_img = np.zeros_like(img)
    B_img = np.zeros_like(img)

    # Assign values - creating "Red channel only" images
    # R_img should be displayed as RGB, so we put the R component in the first channel (index 0)
    # But wait, if R_img is an RGB image for matplotlib, Red is index 0.
    # The original R channel from cv2.split is the Red intensity.
    # So R_img[:, :, 0] = R is correct for visualizing Red color.
    R_img[:, :, 0] = R # Canal Rouge
    G_img[:, :, 1] = G # Canal Vert
    B_img[:, :, 2] = B # Canal Bleu

    # Display using subplots
    figure, plot = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    plot[0].imshow(R_img)
    plot[0].set_title('canal rouge')
    plot[0].axis('off')
    
    plot[1].imshow(G_img)
    plot[1].set_title('canal vert')
    plot[1].axis('off')
    
    plot[2].imshow(B_img)
    plot[2].set_title('canal bleu')
    plot[2].axis('off')
    
    plt.show()
