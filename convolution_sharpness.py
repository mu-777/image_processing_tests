#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/


import cv2
import numpy as np
import time

CASCADE_PATH = "./cascade/lbpcascade_animeface.xml"

IN_IMG_PATHS = ["./test_imgs/face_detecting" + str(i + 1) + ".png" for i in range(9)]
OVERLAY_IMG_PATH = "./test_imgs/face_up5.jpg"
OUT_IMG_PATH = "./test_imgs/face_detecting_out.png"


def check_img(img):
    cv2.imshow('a', img)
    cv2.waitKey(0)


def smooth(img, ksize=(5, 5), sigma=0):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # smooth_img = cv2.GaussianBlur(gray_img, ksize, sigma)
    # return cv2.cvtColor(smooth_img, cv2.COLOR_GRAY2BGR)
    return cv2.blur(img, ksize)


def sharpen(img, k=1.0):
    k = float(k)
    kernel = np.array([[-k / 9.0, -k / 9.0, -k / 9.0],
                       [-k / 9.0, 1 + 8 * k / 9.0, -k / 9.0],
                       [-k / 9.0, -k / 9.0, -k / 9.0]])
    desired_depth = -1
    return cv2.filter2D(img, desired_depth, kernel)


def main(in_img_path):
    rgb_img = cv2.imread(in_img_path)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY),
                                     scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(faces) <= 0:
        return

    for (x, y, w, h) in faces:
        face_img = rgb_img[y:y + h, x:x + w]
        check_img(face_img)
        smooth_img = smooth(face_img)
        check_img(smooth_img)
        diff_img = cv2.absdiff(face_img, smooth_img)
        check_img(diff_img)


# --------------------------------------------
if __name__ == '__main__':
    for img_path in IN_IMG_PATHS:
        main(img_path)
