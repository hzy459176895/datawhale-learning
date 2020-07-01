
"""
haar特征检测人脸（opencv）
"""

import cv2
import numpy as np
haar_front_face_xml = 'xml_file/haarcascade_frontalface_default.xml'
haar_eye_xml = 'xml_file/haarcascade_eye.xml'

# 1.静态图像中的⼈脸检测
def StaticDetect(filename):
     # 创建⼀个级联分类器 加载⼀个 .xml 分类器⽂件. 它既可以是Haar特征也可以是LBP特征的分类器.
     face_cascade = cv2.CascadeClassifier(haar_front_face_xml)
     # 加载图像
     img = cv2.imread(filename)
     # 转换为灰度图
     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     # 进⾏⼈脸检测，传⼊scaleFactor，minNegihbors，分别表示⼈脸检测过程中每次迭代时图像的压缩率以及
     # 每个⼈脸矩形保留近似数⽬的最⼩值
     # 返回⼈脸矩形数组
     faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
     for (x, y, w, h) in faces:
         # 在原图像上绘制矩形
         img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
     cv2.namedWindow('Face Detected！')
     cv2.imshow('Face Detected！', img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


# 视频中的⼈脸检测
def DynamicDetect():
     '''
     打开摄像头，读取帧，检测帧中的⼈脸，扫描检测到的⼈脸中的眼睛，对⼈脸绘制蓝⾊的矩形框，对⼈眼绘制绿⾊的矩形框
     '''
     # 创建⼀个级联分类器 加载⼀个 .xml 分类器⽂件. 它既可以是Haar特征也可以是LBP特征的分类器.
     face_cascade = cv2.CascadeClassifier(haar_front_face_xml)
     eye_cascade = cv2.CascadeClassifier(haar_eye_xml)
     # 打开摄像头
     camera = cv2.VideoCapture(0)
     cv2.namedWindow('Dynamic')
     while True:
         # 读取⼀帧图像
         ret, frame = camera.read()
         # 判断图⽚读取成功？
         if ret:
             gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # ⼈脸检测
         faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
         for (x, y, w, h) in faces:
             # 在原图像上绘制矩形
             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             roi_gray = gray_img[y:y + h, x:x + w]
             # 眼睛检测
             eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
             for (ex, ey, ew, eh) in eyes:
                 cv2.rectangle(frame, (ex + x, ey + y), (x + ex + ew, y + ey + eh), (0, 255, 0),2)
         cv2.imshow('Dynamic', frame)
         # 如果按下q键则退出
         if cv2.waitKey(100) & 0xff == ord('q'):
             break
     camera.release()
     cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = 'data/mayun.jpg'
    # StaticDetect(filename)
    DynamicDetect()
