import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Converting BGR color to RGB color format
    # Essential because OpenCV uses BGR but Matplotlib uses RGB
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using plt
    plt.imshow(RGB_img)
    
    # Show without the axes
    plt.axis('off')
    
    # Put a title
    plt.title('Image')
    plt.show()
