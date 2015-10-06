#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/


import cv2
import numpy as np
import time

# カスケード分類器の特徴量を取得する
cascade_path = "./cascade/lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_path)

in_img_path = "./test_imgs/face_detecting.png"
over_img_path = "./test_imgs/face_up5.jpg"
out_img_path = "./test_imgs/face_detecting_out.png"
img = cv2.imread(in_img_path)
over_img = cv2.imread(over_img_path)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

diff_threshold = 6
overlay_color = (0, 187, 254)
rect_color = (0, 0, 0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

if len(faces) > 0:
    # 検出した顔を囲む矩形の作成
    for (x, y, w, h) in faces:
        over_img_temp = np.zeros((h, w, 3), np.uint8)
        center_color = hsv_img[y + h / 2.0, x + w / 2.0]
        for i, j in [(i, j) for i in range(h) for j in range(w)]:
            # cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, thickness=7)
            color = hsv_img[y + i, x + j]
            if any([abs(diff) < diff_threshold for diff in center_color - color]):
                over_img_temp[i, j] = (255, 255, 255)
        contours = cv2.findContours(cv2.cvtColor(over_img_temp, cv2.COLOR_BGR2GRAY),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours[0], 0, overlay_color, thickness=5)

cv2.imwrite(out_img_path, img)