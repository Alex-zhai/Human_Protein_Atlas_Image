# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/26 11:37

import cv2
import numpy as np

img = cv2.imread('299_0_596d87daN9e638068.jpg')
print(img.shape)
fliplr = np.flipud(img)
print(fliplr.shape)
cv2.imwrite("das.jpg", fliplr)



def rotate_image(im, angle):
    if angle % 90 == 0:
        angle = angle % 360
        if angle == 0:
            return im
        elif angle == 90:
            return im.transpose((1, 0, 2))[:, ::-1, :]
        elif angle == 180:
            return im[::-1, ::-1, :]
        elif angle == 270:
            return im.transpose((1, 0, 2))[::-1, :, :]


img_270 = rotate_image(img, 180)
cv2.imwrite("270.jpg", img_270)


