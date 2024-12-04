# Multiple_Scale _Retinex(M SR)
import cv2
import numpy as np

def multi_scale_retinex(img, scales, weights):

    img = img.astype(np.float32) + 1e-8  # Avoid the zero problem of logarithms
    retinex = np.zeros_like(img) # build zero matrixs like img

    # processing for each scale
    for sigma, weight in zip(scales, weights):
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += weight * (np.log(img) - np.log(blur))

    # normalization
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)

if __name__ == "__main__":
    image_path = "MSR.png"
    img = cv2.imread(image_path)

    # scales = [15, 80, 250]  # Gaussian kernal 1
    # scales = [5, 30, 100]  # Gaussian kernal 2
    scales = [10, 50, 200, 300, 400]  # Gaussian kernal 3
    weights = [1/5, 1/5, 1/5, 1/5, 1/5]  # weights

    if len(img.shape) == 3: # For RGB image
        b, g, r = cv2.split(img)
        b_msr = multi_scale_retinex(b, scales, weights)
        g_msr = multi_scale_retinex(g, scales, weights)
        r_msr = multi_scale_retinex(r, scales, weights)
        result = cv2.merge([b_msr, g_msr, r_msr])
    else:
        result = multi_scale_retinex(img, scales, weights)

    cv2.imshow("Original Image", img)
    cv2.imshow("MSR Result", result)
    cv2.imwrite("MSR_Result.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
