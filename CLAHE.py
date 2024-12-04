"""
利用OpenCV算法库实现CLAHE(限制对比度自适应直方图均衡化)算法 
"""
import cv2 as cv
import numpy as np

# 对单一彩色图片进行直方图均衡化操作
def single_img_color_hist_equalization(file_path, new_file_path):
    """
    file_path: 原始图片存放路径
    new_file_path: 新图片存放路径
    """
    # 读取彩色图片
    test = cv.imread(file_path, cv.IMREAD_COLOR)  # 读取彩色图像
    if test is None:
        print(f"Error: Could not open or find the image {file_path}")
        return

    # 分离BGR三个通道
    (B, G, R) = cv.split(test)

    # 对每个通道应用直方图均衡化
    equalized_B = cv.equalizeHist(B)
    equalized_G = cv.equalizeHist(G)
    equalized_R = cv.equalizeHist(R)

    # 合并均衡化后的通道
    equalized_image = cv.merge((equalized_B, equalized_G, equalized_R))

    # 保存均衡化后的图片
    cv.imwrite(new_file_path, equalized_image)


# 对单一彩色图片进行CLAHE操作
def single_img_color_clahe(file_path, new_file_path):
    """
    file_path: 原始图片存放路径
    new_file_path: 新图片存放路径
    """
    # 读取彩色图片
    test = cv.imread(file_path, cv.IMREAD_COLOR)  # 读取彩色图像
    if test is None:
        print(f"Error: Could not open or find the image {file_path}")
        return

    # 创建CLAHE对象
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 设定参数，clipLimit控制对比度增强强度

    # 分离BGR三个通道
    (B, G, R) = cv.split(test)

    # 对每个通道应用CLAHE
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)

    # 合并均衡化后的通道
    clahe_image = cv.merge((clahe_B, clahe_G, clahe_R))

    # 保存处理后的图片
    cv.imwrite(new_file_path, clahe_image)


if __name__ == "__main__":
    single_img_color_hist_equalization(file_path='./img.png',
                                       new_file_path="./img_basic.png")

    single_img_color_clahe(file_path='./img.png',
                                        new_file_path="./img_clahe.png")