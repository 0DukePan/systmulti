import cv2

# Read the image
# Ensure the image 'cablecar.bmp' is in the same directory
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image 'cablecar.bmp'. Assurez-vous qu'elle est dans le même dossier.")
else:
    print(f"Type de l'objet image: {type(img)}")
    
    # Get the image height, width, and channels
    # shape returns a tuple (height, width, channels)
    height, width, channels = img.shape
    print(f'height: {height} / width: {width} / channels: {channels}')
    
    # Create a window with a title "Image" to display the image
    cv2.imshow("Image", img)
    
    print("Appuyez sur une touche pour fermer la fenêtre...")
    # Hold the screen until the user closes it.
    cv2.waitKey(0)
    
    # Destroy the window
    cv2.destroyAllWindows()
