#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/


import cv2
import numpy as np
import time
from PIL import Image

CASCADE_PATH = "./cascade/lbpcascade_animeface.xml"

IN_IMG_PATHS = ["./test_imgs/face_detecting" + str(i + 1) + ".png" for i in range(9)]
OVERLAY_IMG_PATH = "./test_imgs/face_up3.jpg"
OUT_IMG_PATH = "./test_imgs/face_detecting_out.png"


def check_img(img, title=None):
    t = title if title is not None else 'a'
    cv2.imshow(t, img)
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


def get_alphachannel(face_img):
    (h, w) = face_img.shape[:2]

    bilat_blur_img = cv2.bilateralFilter(face_img, 50, 70, 30)
    # check_img(bilat_blur_img, 'bilat_blur')

    # http://opencv.jp/opencv-2svn/cpp/miscellaneous_image_transformations.html#cv-floodfill
    cv2.floodFill(bilat_blur_img, None,
                  (int(h / 2.0), int(w / 2.0)), (0, 0, 0),
                  loDiff=(3, 3, 3), upDiff=(5, 5, 5))
    # check_img(bilat_blur_img, 'bilat_blur')

    ret, th_img = cv2.threshold(bilat_blur_img, 1, 255, cv2.THRESH_BINARY_INV)
    # check_img(th_img, 'th')

    dilated_img = cv2.dilate(th_img,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                             iterations=5)
    # check_img(dilated_img, 'dilated')

    eroded_img = cv2.erode(dilated_img,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                           iterations=15)
    # check_img(eroded_img, 'eroded')
    return cv2.cvtColor(eroded_img, cv2.COLOR_BGR2GRAY)


def cv2pil(cv_img):
    return Image.fromarray(cv_img[::-1, :, ::-1])


def fill_void(src_img):
    contours, hierarchy = cv2.findContours(src_img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contours.reverse()
    for i, contour in enumerate(contours):
        cv2.drawContours(src_img, [contour], -1, (255, 255, 255), thickness=-1)
    return src_img


def main(in_img_path):
    rgb_img = cv2.imread(in_img_path)
    overlay_img = cv2.imread(OVERLAY_IMG_PATH, -1)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    faces = cascade.detectMultiScale(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY),
                                     scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(faces) <= 0:
        return

    for (x, y, w, h) in faces:
        face_img = rgb_img[y:y + h, x:x + w]
        if w < 100:
            continue

        alpha_channel = get_alphachannel(face_img)
        check_img(alpha_channel)
        alpha_channel = fill_void(alpha_channel)

        resized_overlay_img = cv2.resize(overlay_img, tuple((w, h)))
        mask_img = cv2.bitwise_and(resized_overlay_img, resized_overlay_img,
                                   mask=alpha_channel)
        check_img(mask_img)
        for i, j in [(i, j) for i in range(h) for j in range(w)]:
            if any(mask_img[i, j]):
                rgb_img[y + i, x + j] = mask_img[i, j]

    check_img(rgb_img)


# --------------------------------------------
if __name__ == '__main__':
    for in_img_path in IN_IMG_PATHS:
        main(in_img_path)
