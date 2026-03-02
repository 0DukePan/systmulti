import cv2
import numpy as np
import random

# ==========================================
# 2- Custom Convolution Function
# ==========================================
def convolution(pad_img, kernel):
    """Manual convolution as defined in the TP"""
    # Calculate padding size based on kernel dimensions
    p = int(kernel.shape[0] / 2)
    pheight, pwidth = pad_img.shape

    # Initialize output image with zeros
    img_conv = np.zeros(pad_img.shape)

    # Iterate through the image pixels
    for i in range(p, pheight - p):
        for j in range(p, pwidth - p):
            # Extract the Region of Interest (ROI) matching kernel size
            roi = pad_img[i-p:i+p+1, j-p:j+p+1]
            # Perform element-wise multiplication and sum
            img_conv[i, j] = np.sum(kernel * roi)

    # Remove the padding to return to original image dimensions
    img_conv = img_conv[p:-p, p:-p]
    return img_conv

# ==========================================
# 1 & 3- Main Program Execution
# ==========================================

# Step 1: Read the color image and convert to grayscale
# Adjusted path: "../" looks in the parent directory (TP1) for the image
gray_img = cv2.imread('../cablecar.bmp', cv2.IMREAD_GRAYSCALE)

if gray_img is None:
    print("Error: 'cablecar.bmp' not found in the parent directory.")
else:
    # --- PART 1 & 2 Comparison ---
    # Define a blurring kernel (Kernel 1)
    kernel1 = np.ones((5, 5)) / 30

    # Pad the image manually for the custom function
    p = int(kernel1.shape[0] / 2)
    pad_img = np.zeros((gray_img.shape[0] + 2*p, gray_img.shape[1] + 2*p))
    pad_img[p:-p, p:-p] = gray_img  # Place original image in center

    # Apply both methods
    img_conv_cv2 = cv2.filter2D(src=gray_img, ddepth=cv2.CV_64F, kernel=kernel1)
    img_conv_custom = convolution(pad_img, kernel1)

    # --- PART 3: 16-Block Random Filter Exercise ---
    # Define the required kernels
    k_laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    k_emboss    = np.array([[-2, -1,  0], [-1, 1,  1], [ 0, 1,  2]])
    k_sharpen   = np.array([[ 0, -1,  0], [-1, 5, -1], [ 0, -1, 0]])
    k_sobel_v   = np.array([[-1,  0,  1], [-2, 0,  2], [-1, 0,  1]])
    k_gaussian  = np.array([[ 1,  2,  1], [ 2, 4,  2], [ 1, 2,  1]]) / 16.0

    kernels_list = [k_laplacian, k_emboss, k_sharpen, k_sobel_v, k_gaussian]

    height, width = gray_img.shape
    # Diviser l'image en 16 blocs de même dimension
    h_b, w_b = height // 4, width // 4
    final_img = np.zeros_like(gray_img)

    for i in range(4):
        for j in range(4):
            # Define block boundaries
            y1, y2 = i * h_b, (i + 1) * h_b
            x1, x2 = j * w_b, (j + 1) * w_b

            # Select a random kernel and apply it to the block
            block = gray_img[y1:y2, x1:x2]
            chosen_k = random.choice(kernels_list)  # noyau choisi aléatoirement
            # ddepth=-1 keeps the output the same depth as the source
            final_img[y1:y2, x1:x2] = cv2.filter2D(src=block, ddepth=-1, kernel=chosen_k)

    # Display results
    # Regrouper les 16 blocs filtrés dans une seule image et l'afficher.
    cv2.imshow('1. Original Grayscale', gray_img)
    cv2.imshow('2. Custom Conv (Blur)', np.uint8(np.abs(img_conv_custom)))
    cv2.imshow('3. 16 Blocks Random Filters', final_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()