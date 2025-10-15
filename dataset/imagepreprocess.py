# @Time : 2022-06-27 16:32
# @Author : Wang Zhen
# @Email : frozenzhencola@163.com
# @File : imagepreprocess.py
# @Project : GPR-R2-0625
import cv2
import numpy as np
def get_image_rotation(image,rotation):
    #通用写法，即使传入的是三通道图片依然不会出错
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    #得到旋转矩阵，第一个参数为旋转中心，第二个参数为旋转角度，第三个参数为旋转之前原图像缩放比例
    M = cv2.getRotationMatrix2D(center, -rotation, 1)
    #进行仿射变换，第一个参数图像，第二个参数是旋转矩阵，第三个参数是变换之后的图像大小
    image_rotation = cv2.warpAffine(image, M, (width, height))
    return image_rotation
def satellite_image_preprocess(img):
    img = img[60:460,50:450]
    # img = hisEqulColor(img)
    return img

def uav_image_preprocess(img):
    img0_r = get_image_rotation(img, -14)
    img = img0_r[50:450, 50:450]
    # img0_r = hisEqulColor(img0_r)
    return img
def hisEqulColor(img):
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img