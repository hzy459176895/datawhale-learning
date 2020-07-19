
"""
提取图像hog特征 + svm分类
实现图像中的行人检测...
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# # gramma变换 图像预处理 案例 #########################
# img = cv2.imread('data/xingren.jpg', 0)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img2 = np.power(img/float(np.max(img)),1/2.2)
# plt.imshow(img2)
# plt.axis('off')
# plt.show()


# # 计算图像梯度 案例 #######################
# # Read image
# img = cv2.imread('data/xingren.jpg')
# img = np.float32(img) / 255.0  # 归一化
# # 计算x和y方向的梯度
# gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
# gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
# # 计算合梯度的幅值和方向（角度）
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
# print('ok')


# 行人检测案例 ##############################
if __name__ == '__main__':
    src = cv.imread('data/xingren.jpg')
    cv.imshow("input", src)

    hog = cv.HOGDescriptor()

    # opencv的 默认的 hog+svm行人检测器
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    # Detect people in the image
    (rects, weights) = hog.detectMultiScale(src,
                                            winStride=(2, 4),
                                            padding=(8, 8),
                                            scale=1.2,
                                            useMeanshiftGrouping=False)
    for (x, y, w, h) in rects:
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("hog-detector", src)
    cv.imwrite("hog-detector.jpg", src)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # # hog特征可视化 ############################
    # from skimage import feature, exposure
    # import cv2
    #
    # image = cv2.imread('sp_g.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
    #                             cells_per_block=(2, 4), visualise=True)
    #
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow('img', image)
    # cv2.namedWindow("hog", cv2.WINDOW_NORMAL)
    # cv2.imshow('hog', hog_image_rescaled)
    # cv2.waitKey(0)