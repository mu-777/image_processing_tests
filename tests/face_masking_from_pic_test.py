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
rgb_img = cv2.imread(in_img_path)
over_img = cv2.imread(over_img_path)
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

overlay_color = (0, 187, 254)
rect_color = (0, 0, 0)

faces = cascade.detectMultiScale(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY),
                                 scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))


def cutoff_rgb(x, y, w, h):
    diff_threshold = 20
    over_img_temp = np.zeros((h, w, 3), np.uint8)
    center_color = rgb_img[y + h / 2.0, x + w / 2.0]
    for i, j in [(i, j) for i in range(h) for j in range(w)]:
        color = rgb_img[y + i, x + j]
        if all([abs(diff) < diff_threshold for diff in center_color - color]):
            over_img_temp[i, j] = rgb_img[i + y, j + x]
    return over_img_temp


def cutoff_hsv(x, y, w, h):
    diff_threshold = 6
    over_img_temp = np.zeros((h, w, 3), np.uint8)
    center_color = hsv_img[y + h / 2.0, x + w / 2.0]
    for i, j in [(i, j) for i in range(h) for j in range(w)]:
        color = hsv_img[y + i, x + j]
        if all([abs(diff) < diff_threshold for diff in center_color - color]):
            over_img_temp[i, j] = rgb_img[i + y, j + x]
    return over_img_temp


def check_img(img):
    cv2.imshow('a', img)
    cv2.waitKey(0)


if len(faces) > 0:
    # 検出した顔を囲む矩形の作成
    for (x, y, w, h) in faces:
        print(w, h)
        over_img_temp = rgb_img[y:y + h, x:x + w]
        gray = cv2.cvtColor(over_img_temp, cv2.COLOR_BGR2GRAY)
        gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        # edge_img = cv2.Canny(gray_smooth, 1000, 1500, apertureSize=5)
        edge_img = cv2.Canny(gray_smooth, 1600, 1600, apertureSize=5)
        check_img(edge_img)
        dilated_img = cv2.dilate(edge_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=3)
        check_img(dilated_img)
        cv2.imwrite('./'+str(x)+'dilated_img.jpg', dilated_img)
        contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        c_len = len(contours)
        for i, contour in enumerate(contours):
            cv2.drawContours(over_img_temp, [contour], -1, (0, 255 * float(i) / c_len, 0), thickness=-1)
        check_img(over_img_temp)
        cv2.imwrite('./'+str(x)+'over_img.jpg', over_img_temp)


        # contour_img = over_img_temp.copy()
        # for i, contour in enumerate(contours):
        # arclen = cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, 0.02 * arclen, True)
        # cv2.drawContours(contour_img, [approx], -1,
        #                      (0, 0, 255 * (1 - float(i) / len(contours))), 2)
        # check_img(contour_img)


        # contour = reduce(lambda c1, c2: np.r_[c1, c2], contours)
        # cv2.fillConvexPoly(over_img_temp, contour, (255, 0, 0))

        # for contour in contours:
        # if len(contour) > 10:
        # box = cv2.fitEllipse(contour)
        # cv2.ellipse(over_img_temp, box, (255, 255, 0), 2)
        # check_img(over_img_temp)

        # over_img_temp = cutoff_rgb(x, y, w, h)
        # over_img_temp = cutoff_hsv(x, y, w, h)
        # kernel_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        # kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # ret, binary_img = cv2.threshold(over_img_temp, 130, 255, cv2.THRESH_BINARY)
        # first = cv2.dilate(binary_img, kernel_l)
        # second = cv2.erode(first, kernel_s, iterations=5)
        # first = cv2.dilate(binary_img, kernel_l)
        # second = cv2.erode(first, kernel_s, iterations=5)
        # check_img(binary_img)
        # check_img(first)
        # check_img(second)

        # gray = cv2.cvtColor(over_img_temp, cv2.COLOR_BGR2GRAY)
        # gray_smooth = cv2.GaussianBlur(gray, (31, 31), 0)
        # ret, th1 = cv2.threshold(gray_smooth, 130, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(over_img_temp, contours, 0, overlay_color, thickness=5)
        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 187, 254), thickness=7)

# cv2.imwrite(out_img_path, rgb_img)