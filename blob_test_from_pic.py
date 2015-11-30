#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/
# Blob
# http://www.learnopencv.com/blob-detection-using-opencv-python-c/

import cv2
import numpy as np
import time

# カスケード分類器の特徴量を取得する
CASCADE_PATH = "./cascade/lbpcascade_animeface.xml"
IN_IMG_PATHS = ["./test_imgs/face_detecting" + str(i + 1) + ".png" for i in range(9)]
OVERLAY_IMG_PATH = "./test_imgs/face_up5.jpg"
OUT_IMG_PATH = "./test_imgs/face_detecting_out.png"


overlay_color = (0, 187, 254)
rect_color = (0, 0, 0)


def check_img(img):
    cv2.imshow('a', img)
    cv2.waitKey(0)


def cutoff_hsv(src_img, diff_threshold=6):
    (h, w) = src_img.shape[:2]
    hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    ret_img = np.zeros((h, w, 3), np.uint8)
    (c_h, c_s, c_v) = hsv_img[h / 2.0, w / 2.0]
    for i, j in [(i, j) for i in range(h) for j in range(w)]:
        (h, s, v) = hsv_img[i, j]
        if abs(c_h - h) < diff_threshold:
            ret_img[i, j] = src_img[i, j]
    return ret_img


def cutoff_rgb(src_img, diff_threshold=20):
    (h, w) = src_img.shape[:2]
    ret_img = np.zeros((h, w, 3), np.uint8)
    center_color = src_img[h / 2.0, w / 2.0]
    for i, j in [(i, j) for i in range(h) for j in range(w)]:
        color = src_img[i, j]
        if all([abs(diff) < diff_threshold for diff in center_color - color]):
            ret_img[i, j] = src_img[i, j]
    return ret_img



def main(in_img_path):
    rgb_img = cv2.imread(in_img_path)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY),
                                     scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector(params)

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
            keypoints = detector.detect(dilated_img)
            im_with_keypoints = cv2.drawKeypoints(dilated_img, keypoints,
                                                  np.array([]), (0,0,255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            check_img(im_with_keypoints)
            check_img(over_img_temp)

# --------------------------------------------
if __name__ == '__main__':
    for img_path in IN_IMG_PATHS:
        main(img_path)
