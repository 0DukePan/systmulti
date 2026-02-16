import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Convert to RGB for display
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Add padding
    # pad_width=((top, bottom), (left, right), (channels_before, channels_after))
    # ((10, 10), (10, 10), (0, 0)) Adds 10 pixels border around the image
    # mode='constant', constant_values=0 adds a black border
    padded_image = np.pad(RGB_img, pad_width=((10, 10), (10, 10), (0, 0)), mode='constant', constant_values=0)
    
    figure, plot = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    plot[0].imshow(RGB_img)
    plot[0].set_title('l’image originale')
    plot[0].axis('off')
    
    plot[1].imshow(padded_image)
    plot[1].set_title('apres le padding')
    plot[1].axis('off')
    
    plt.show()
