import cv2
import numpy as np

# calculating dark channel
def dark_channel(img, size=15):
    min_img = np.min(img, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    #dc = np.clip(dc_img, 0, 1)
    #cv2.imwrite('img/haze/results/8_darkchannel.png', np.uint8(dc * 255))
    return dc_img

# estimate global atmospheric light A
def get_atmo(img, percent=0.001):
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)

# estimate transmission map t
def get_trans(img, atom, w=0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t

# guided filtering
def guided_filter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """

    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * i + mean_b
    return q

def DCP(image_path, Gamma):
    im = cv2.imread(image_path)
    #cv2.imshow('original+image', im)
    #cv2.waitKey(0)
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    atom = get_atmo(img)
    print(atom)
    #atom = 0.8113288519594382
    trans = get_trans(img, atom)
    #cv2.imshow('p', trans)
    #cv2.waitKey(0)

    #DCP_VO
    """
    pre_result = np.empty_like(img)
    for i in range(3):
        pre_result[:, :, i] = (img[:, :, i] - atom) / trans + atom
    pre_result = np.clip(pre_result, 0, 1)
    cv2.imshow('pre_result', pre_result)
    cv2.waitKey(0)
    #cv2.imwrite('img/haze/results/DCP_v0.png', np.uint8(pre_result * 255))
    """

    #DCP_v1
    """
    mid_result = np.empty_like(img)
    mid_trans = cv2.max(trans, 0.2)
    for i in range(3):
        mid_result[:, :, i] = (img[:, :, i] - atom) / mid_trans + atom
    mid_result = np.clip(mid_result, 0, 1)
    cv2.imshow('mid_result', mid_result)
    cv2.waitKey(0)
    #cv2.imwrite('img/haze/results/DCP_v1.png', np.uint8(mid_result * 255))
    """

    #DCP_v3
    trans_guided = guided_filter(trans, img_gray, 81, 0.0001)
    #cv2.imshow('trans', trans)
    #cv2.imwrite('img/haze/results/trans_81.png', np.uint8(trans * 255))
    #cv2.imshow('trans_guided', trans_guided)
    #cv2.imwrite('img/haze/results/trans_guided_81.png', np.uint8(trans_guided * 255))
    #cv2.waitKey(0)
    trans_guided = cv2.max(trans_guided, 0.25)
    #cv2.imshow("trans_guided", trans_guided)
    #cv2.waitKey(0)
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
    result = np.clip(result, 0, 1)
    if Gamma:
        result = result ** (np.log(0.5) / np.log(result.mean()))
    cv2.imshow("result", result)
    cv2.waitKey(0)
    #cv2.imwrite('img/haze/results/DCP_v4.png', np.uint8(result * 255))

if __name__ == '__main__':
    DCP('img/haze/1baseline/8.png', True)
