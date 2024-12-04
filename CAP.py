import cv2  # 导入OpenCV库，用于图像处理和显示
import numpy as np  # 导入NumPy库，用于高效的数组和矩阵运算


# 定义去雾函数
def color_attenuation_prior_dehazing(image, omega=0.9, t0=0.7, t1=0.9):
    """
    基于颜色衰减先验的去雾算法实现。
    （函数说明省略，与原文相同）
    """
    if image is None:
        raise ValueError("输入图像为空，请检查图像路径。")

    img = image.astype(np.float32) / 255.0
    B = img[:, :, 0]
    transmission = np.clip(omega * (1 - B), t0, t1)
    img_out = (img - transmission[..., np.newaxis]) / (1 - transmission[..., np.newaxis] + 1e-5)
    img_out = np.clip(img_out, 0, 1)
    return (img_out * 255).astype(np.uint8)


# 读取输入图像
image_path = 'D:/PyCharm/project/figure/test1.jpg'  # 指定图像路径
image = cv2.imread(image_path)  # 使用OpenCV读取图像

# 检查图像是否成功加载
if image is None:
    print(f"错误：无法加载图像 {image_path}")
else:
    # 调用去雾函数处理图像
    dehazed_image = color_attenuation_prior_dehazing(image)

    # 使用OpenCV显示原始图像和去雾后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Dehazed Image', dehazed_image)

    # 保存去雾后的图像
    output_path = 'D:/PyCharm/project/figure/dehazed_test1.jpg'  # 指定输出图像路径
    cv2.imwrite(output_path, dehazed_image)  # 保存图像
    print(f"去雾后的图像已保存为 {output_path}")

    # 等待用户按键操作，然后关闭所有图像窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 销毁所有OpenCV创建的窗口