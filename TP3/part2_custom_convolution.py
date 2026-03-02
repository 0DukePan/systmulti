import cv2
import numpy as np

# ==========================================
# PART 2 - Custom (Manual) Convolution
# ==========================================

def convolution(pad_img, kernel):
    """Manual convolution: slides the kernel over every pixel and computes
    the element-wise product with the local Region of Interest (ROI)."""
    p = int(kernel.shape[0] / 2)
    pheight, pwidth = pad_img.shape

    img_conv = np.zeros(pad_img.shape)

    for i in range(p, pheight - p):
        for j in range(p, pwidth - p):
            roi = pad_img[i-p:i+p+1, j-p:j+p+1]
            img_conv[i, j] = np.sum(kernel * roi)

    # Remove padding to restore original dimensions
    img_conv = img_conv[p:-p, p:-p]
    return img_conv


def pad_and_convolve(gray_img, kernel):
    """Helper: pads the image and runs the custom convolution."""
    p = int(kernel.shape[0] / 2)
    pad_img = np.zeros((gray_img.shape[0] + 2*p, gray_img.shape[1] + 2*p))
    pad_img[p:-p, p:-p] = gray_img
    return convolution(pad_img, kernel)


gray_img = cv2.imread('../cablecar.bmp', cv2.IMREAD_GRAYSCALE)

if gray_img is None:
    print("Error: 'cablecar.bmp' not found in the parent directory.")
else:
    # Kernel 1 - Blur (averaging)
    kernel1 = np.ones((5, 5)) / 30

    # Kernel 2 - Sharpen
    kernel2 = np.array([[ -1, -1,  -1],
                        [-1,  8, -1],
                        [ -1, -1,  -1]], dtype=np.float64)

    # Apply both kernels through the custom convolution
    result_k1 = pad_and_convolve(gray_img, kernel1)
    result_k2 = pad_and_convolve(gray_img, kernel2)

    # Display results
    cv2.imshow('Original Grayscale', gray_img)
    cv2.imshow('Part 2 - Kernel 1: Blur (custom conv)',    np.uint8(np.abs(result_k1)))
    cv2.imshow('Part 2 - Kernel 2: Sharpen (custom conv)', np.uint8(np.clip(result_k2, 0, 255)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
