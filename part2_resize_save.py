import cv2

# Read the image
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Resize an image
    # Note: shape is (height, width, channels), so shape[1] is width, shape[0] is height
    # Resizing to half the original dimensions
    resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    # Save the image
    cv2.imwrite("resized.png", resized_img)
    print("Image redimensionnée et sauvegardée sous 'resized.png'.")
