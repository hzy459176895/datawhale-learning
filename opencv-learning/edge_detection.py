
"""
sobel算子边缘检测
canny算子边缘检测
"""

# # sobel 边缘检测 ##############################
# """
# cv2.Sobel(src, #参数是需要处理的图像；
# 					ddepth, #图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度
# 					dx, #dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
# 					dy[,
# 					dst[, #输出图片
#           ksize[,#Sobel算子的大小，必须为1、3、5、7。
#           scale[, #缩放导数的比例常数，默认情况下没有伸缩系数；
#           delta[, #可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
#           borderType #判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
#           ]]]]])
# """
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# #读图
# img = cv2.imread('data/fangzi_pic.png')
#
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#
# #画图
# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(1,3,2),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(1,3,3),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#
# plt.show()


# canny 边缘检测 #######################################
# -*- coding: utf-8 -*-

"""
cv2.Canny(image,            # 输入原图（必须为单通道图）
          threshold1,
          threshold2,       # 较大的阈值2用于检测图像中明显的边缘
          [, edges[,
          apertureSize[,    # apertureSize：Sobel算子的大小
          L2gradient ]]])   # 参数(布尔值)：
                              true： 使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），
                              false：使用L1范数（直接将两个方向导数的绝对值相加）。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

original_img = cv2.imread('data/fangzi_pic.png', 0)

# canny边缘检测
img1 = cv2.GaussianBlur(original_img,(3,3),0)
canny = cv2.Canny(img1, 50, 150)

# 画图
plt.subplot(1,2,1),plt.imshow(original_img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.show()
