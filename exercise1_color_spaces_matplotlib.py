import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Convert using OpenCV
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original (RGB for correct display)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original (RGB)')
    axes[0].axis('off')

    # Grayscale
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title('Niveau de gris')
    axes[1].axis('off')
    
    # YCrCb
    # Note: Displaying raw YCrCb data as RGB produces "false colors" but visualizes the data content
    axes[2].imshow(img_ycrcb)
    axes[2].set_title('YCrCb')
    axes[2].axis('off')
    
    # HSV
    # Similarly for HSV
    axes[3].imshow(img_hsv)
    axes[3].set_title('HSV')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
