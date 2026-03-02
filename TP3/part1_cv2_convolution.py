import cv2
import numpy as np

# ==========================================
# PART 1 - Convolution using cv2.filter2D
# ==========================================

# Step 1: Read the image and convert to grayscale
gray_img = cv2.imread('../cablecar.bmp', cv2.IMREAD_GRAYSCALE)

if gray_img is None:
    print("Error: 'cablecar.bmp' not found in the parent directory.")
else:
    # Define a blurring kernel (5x5, averaging)
    kernel1 = np.ones((5, 5)) / 30

    # Apply convolution using OpenCV's built-in filter2D
    img_conv_cv2 = cv2.filter2D(src=gray_img, ddepth=cv2.CV_64F, kernel=kernel1)

    # Display results
    cv2.imshow('Original Grayscale', gray_img)
    cv2.imshow('Part 1 - cv2.filter2D (Blur)', np.uint8(np.abs(img_conv_cv2)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
