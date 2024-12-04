# Single_Scale _Retinex(SSR)
import cv2
import numpy as np

def single_scale_retinex(img, sigma):

    img = img.astype(np.float32) + 1e-8  # Avoid the zero problem of logarithms
    blur = cv2.GaussianBlur(img, (0, 0), sigma)# Gaussian Blur

    # SSR equation = log(I) - log(G) and then normalization
    retinex = np.log(img) - np.log(blur)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)

    return retinex.astype(np.uint8)

if __name__ == "__main__":
    image_path = "SSR.png"
    img = cv2.imread(image_path)

    sigma = 300 # Gausian kernal
    if len(img.shape) == 3: # For RGB image
        b, g, r = cv2.split(img)
        b_ssr = single_scale_retinex(b, sigma)
        g_ssr = single_scale_retinex(g, sigma)
        r_ssr = single_scale_retinex(r, sigma)
        result = cv2.merge([b_ssr, g_ssr, r_ssr])
    else:
        result = single_scale_retinex(img, sigma)

    cv2.imshow("Original Image", img)
    cv2.imshow("SSR Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('SSR_result.png', result)
