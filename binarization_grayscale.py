import sys
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter
import numpy as np

def process_image(image_path):
    # 打开图片
    image = Image.open(image_path)

    # 灰度化处理
    grayscale_image = image.convert("L")
    grayscale_image.save("grayscale_image.jpg")
    print("灰度化图片已保存为 grayscale_image.jpg")

    # 二值化处理
    threshold = 128
    binary_image = grayscale_image.point(lambda p: p > threshold and 255)
    binary_image.save("binary_image.jpg")
    print("二值化图片已保存为 binary_image.jpg")

    # 中值滤波
    grayscale_array = np.array(grayscale_image)
    median_filtered = median_filter(grayscale_array, size=5)
    Image.fromarray(median_filtered).save("median_filtered.jpg")
    print("中值滤波图片已保存为 median_filtered.jpg")

    # 高斯滤波
    gaussian_filtered = gaussian_filter(grayscale_array, sigma=1)
    Image.fromarray(gaussian_filtered).save("gaussian_filtered.jpg")
    print("高斯滤波图片已保存为 gaussian_filtered.jpg")

# 替换为你的图片路径
image_path = "./dataset/test/calendula_flower/calendula_flower_12.jpg"
process_image(image_path)
