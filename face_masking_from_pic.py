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


def main(i):
    rgb_img = cv2.imread(IN_IMG_PATHS[i])
    overlay_img = cv2.imread(OVERLAY_IMG_PATH)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY),
                                     scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(faces) <= 0:
        return

    for (x, y, w, h) in faces:
        face_img = rgb_img[y:y + h, x:x + w]
        gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        print(w)
        if w > 300:
            gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
        elif w > 200:
            gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        elif w > 100:
            gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        edge_img = cv2.Canny(gray_img, 0, 1500, apertureSize=5)
        check_img(edge_img)
        dilated_img = cv2.dilate(edge_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=76)
        check_img(dilated_img)
        contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        c_len = len(contours)
        for i, contour in enumerate(contours):
            cv2.drawContours(face_img, [contour], -1, (0, 255 * float(i) / c_len, 0), thickness=-1)
        check_img(face_img)
        check_img(edge_img)


# --------------------
for i in range(9):
    main(i)
