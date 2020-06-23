
"""
图像变换  （仿射变换）
"""
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('data/test_pic1.png')


# 图像翻转 ############
# Flipped Horizontally 水平翻转
h_flip = cv.flip(img, 1)
# Flipped Vertically 垂直翻转
v_flip = cv.flip(img, 0)
# Flipped Horizontally & Vertically 水平垂直翻转
hv_flip = cv.flip(img, -1)

plt.figure(figsize=(8,8))

plt.subplot(221)
plt.imshow(img[:,:,::-1])
plt.title('original')

plt.subplot(222)
plt.imshow(h_flip[:,:,::-1])
plt.title('horizontal flip')

plt.subplot(223)
plt.imshow(v_flip[:,:,::-1])
plt.title(' vertical flip')

plt.subplot(224)
plt.imshow(hv_flip[:,:,::-1])
plt.title('h_v flip')
# 调整子图间距
# plt.subplots_adjust(wspace=0.5, hspace=0.1)
plt.subplots_adjust(top=0.8, bottom=0.08, left=0.10, right=0.95, hspace=0,
                    wspace=0.35)
# plt.tight_layout()
plt.show()


# # 平移和旋转  （M是定义平移矩阵 或者 旋转矩阵）##########################
# rows, cols = img.shape[:2]
#
# # 定义平移矩阵，需要是numpy的float32类型
# # x轴平移30，y轴平移20
# M = np.float32([[1, 0, 30], [0, 1, 20]])
# # 用仿射变换实现平移, borderValue是背景设RGB值
# # 参数：图片，平移矩阵，背景大小，颜色值
# img_s = cv.warpAffine(img, M, (300, 300), borderValue=(255, 0, 0))
#
# # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
# M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)  # 以原图为中心，旋转90度，2倍大小
# M1 = cv.getRotationMatrix2D((cols/2,rows/2),45,1)
# M2 = cv.getRotationMatrix2D((cols/2,rows/2),78,1)
# print(M)
#
#
# # 第三个参数：变换后的图像大小
# img_tra = cv.warpAffine(img,M,(300,300))
# img_tra1 = cv.warpAffine(img,M1,(300,300))
# img_tra2 = cv.warpAffine(img,M2,(300,300), borderValue=(0, 0, 255))
#
# plt.figure(figsize=(8,8))
# plt.subplot(221)
# plt.imshow(img[:,:,::-1])
#
# plt.subplot(222)
# plt.imshow(img_s[:,:,::-1])
#
# plt.subplot(223)
# plt.imshow(img_tra[:,:,::-1])
#
# plt.subplot(224)
# plt.imshow(img_tra2[:,:,::-1])
#
# plt.subplots_adjust(top=0.8, bottom=0.08, left=0.10, right=0.95, hspace=0,
#                     wspace=0.35)
# plt.show()
