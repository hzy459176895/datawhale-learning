
"""
harris 角点检测
从图像中检测属于 角点 的点信息
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


# 检测器参数
block_size = 3  # 窗口尺寸大小
sobel_size = 3  # sobel算子尺寸大小
k = 0.06  # 用于计算相应角点R函数的k参数

image = cv.imread('data/fangzi_pic.png')

print(image.shape)
height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]
print("width: %s  height: %s  channels: %s" % (width, height, channels))

gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# modify the data type setting to 32-bit floating point
gray_img = np.float32(gray_img)

# 检测角点
corners_img = cv.cornerHarris(gray_img, block_size, sobel_size, k)

# 膨胀处理，利于更方便的观察角点形态
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dst = cv.dilate(corners_img, kernel)


# 晒出那些不合适的角点，pix大于一定阈值的认为是需要留下的角点
for r in range(height):
    for c in range(width):
        pix = dst[r, c]
        # if pix > 0.05 * dst.max():
        if pix > 0.1 * dst.max():
            cv.circle(image, (c, r), 5, (0, 0, 255), 0)

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

cv.imwrite('data/fangzi_pic_harris_res.jpg', image)
