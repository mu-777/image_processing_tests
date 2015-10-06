#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/


import cv2
import time

# カスケード分類器の特徴量を取得する
cascade_path = "./cascade/lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_path)

in_img_path = "./test_imgs/face_detecting.png"
out_img_path = "./test_imgs/face_detecting_out.png"
img = cv2.imread(in_img_path)

color = (0, 187, 254)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

if len(faces) > 0:
    # 検出した顔を囲む矩形の作成
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=7)

cv2.imwrite(out_img_path, img)