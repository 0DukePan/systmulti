import cv2
import numpy as np

# Read the image
img = cv2.imread('cablecar.bmp')

if img is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Resize for better visualization
    # Reused logic from Part 2
    resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    # Convert from BGR to YCrCb color space: Y is the Luminance component, Cb and Cr chrominance components
    img_ycrcb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YCrCb)
    cv2.imwrite("ycrcb_random.png", img_ycrcb)

    # Convert from BGR to HSV color space
    img_hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv_random.png", img_hsv)

    # Convert an RGB image to grayscale
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    # Concatenate images Horizontally
    # img_gray is 2D, convert back to 3 channels (BGR) to allow concatenation
    img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Concatenate: Gray (re-colored to BGR), YCrCb, HSV
    images = np.concatenate((img_gray_bgr, img_ycrcb, img_hsv), axis=1)
    
    cv2.imshow('Gray, YCrCb and HSV images', images)
    print("Appuyez sur une touche pour fermer la fenêtre...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
