#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import stats
from datetime import datetime
from PIL import Image

OUT_IMG_PATH = "./test_imgs/faces/"
CHECK_IMG_FLAG = False


def check_img(img, title=None, flag=False, output_path=None):
    if flag:
        t = title if title is not None else 'a'
        cv2.namedWindow(t, cv2.WINDOW_NORMAL)
        cv2.imshow(t, img)
        cv2.waitKey(0)
        if output_path is not None:
            cv2.imwrite(output_path + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + title + '.png', img)


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
    check_img(bilat_blur_img, 'bilat_blur', CHECK_IMG_FLAG, OUT_IMG_PATH)

    # http://opencv.jp/opencv-2svn/cpp/miscellaneous_image_transformations.html#cv-floodfill
    cv2.floodFill(bilat_blur_img, None,
                  (int(h / 2.0), int(w / 2.0)), (0, 0, 0),
                  loDiff=(3, 3, 3), upDiff=(5, 5, 5))
    check_img(bilat_blur_img, 'floodFill', CHECK_IMG_FLAG, OUT_IMG_PATH)

    ret, th_img = cv2.threshold(bilat_blur_img, 1, 255, cv2.THRESH_BINARY_INV)
    check_img(th_img, 'th', CHECK_IMG_FLAG, OUT_IMG_PATH)

    dilated_img = cv2.dilate(th_img,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                             iterations=5)
    check_img(dilated_img, 'dilated', CHECK_IMG_FLAG, OUT_IMG_PATH)

    eroded_img = cv2.erode(dilated_img,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                           iterations=15)
    check_img(eroded_img, 'eroded', CHECK_IMG_FLAG, OUT_IMG_PATH)
    return cv2.cvtColor(eroded_img, cv2.COLOR_BGR2GRAY)


def cv2pil(cv_img):
    return Image.fromarray(cv_img[::-1, :, ::-1])


def is_skin(src_img):
    px_size = 11
    center_px = int((px_size + 1) / 2.0)
    range_width = 0
    center_range = range(center_px - range_width - 1, center_px + range_width)

    shrinked_img = cv2.resize(src_img, tuple((px_size, px_size)))
    hsv_img = cv2.cvtColor(shrinked_img, cv2.COLOR_BGR2HSV)

    center_hsv = (0, 0, 0)
    cnt = 0
    for i, j in [(i, j) for i in center_range for j in center_range]:
        cnt = cnt + 1
        center_hsv = center_hsv + hsv_img[i, j]
    if cnt != 0:
        center_hsv = center_hsv / cnt

    # http://d.hatena.ne.jp/mintsu123/20111123/1322065624
    h_flag = 0 <= center_hsv[0] <= 23
    s_flag = center_hsv[1] < 50
    v_flag = 80 < center_hsv[2]
    return h_flag and s_flag and v_flag


def fill_void(src_img):
    contours, hierarchy = cv2.findContours(src_img, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contours.reverse()
    for i, contour in enumerate(contours):
        cv2.drawContours(src_img, [contour], -1, (255, 255, 255), thickness=-1)
    return src_img


def is_front_face(face_img):
    # TODO: UPDATE
    px_size = 11
    shrinked_img = cv2.resize(face_img, tuple((px_size, px_size)))
    check_img(shrinked_img, 'shrinked_img')

    hsv_img = cv2.cvtColor(shrinked_img, cv2.COLOR_BGR2HSV)
    top_hsv = (hsv_img[0, 0] + hsv_img[1, 0]) / 2.0
    bottom_hsv = (hsv_img[0, 1] + hsv_img[1, 1]) / 2.0
    left_hsv = (hsv_img[0, 0] + hsv_img[0, 1]) / 2.0
    right_hsv = (hsv_img[1, 0] + hsv_img[1, 1]) / 2.0
    print(top_hsv[0], bottom_hsv[0], left_hsv[0], right_hsv[0])

    return True


def get_hair_color_hsv(face_img):
    def h_median():
        hair_line = img[hair_line_px, :]
        h_list = [px[0] for px in hair_line]
        median_h = np.median(h_list)
        median_idx = h_list.index(median_h)
        return img[hair_line_px, median_idx]

    def h_mode():
        hair_line = img[hair_line_px, :]
        h_list = [px[0] for px in hair_line]
        mode_h = stats.mode(h_list)[0]
        mode_idx = h_list.index(mode_h)
        return img[hair_line_px, mode_idx]

    def hsv_median():
        hair_line = img[hair_line_px, :]
        ave_color = map(int, tuple(map(np.median, zip(*hair_line))))
        return tuple(ave_color)

    def hsv_mode():
        hair_line = img[hair_line_px, :]
        print(stats.mode(hair_line)[0][0])
        ave_color = map(int, stats.mode(hair_line)[0][0])
        return tuple(ave_color)

    px_size = 11
    center_px = int((px_size + 1) / 2.0)
    hair_line_px = int((px_size + 1) / 6) - 1
    hair_line_px = 0
    face_img = cv2.resize(face_img, tuple((px_size, px_size)))
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

    return h_mode()

def hsv_to_bgr(hsv):
    h, s, v = hsv
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return np.array((int(bgr[0]), int(bgr[1]), int(bgr[2])))


def rgb_to_hsv(rgb):
    r, g, b = rgb
    hsv = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return np.array((int(hsv[0]), int(hsv[1]), int(hsv[2])))


def bgr_to_hsv(bgr):
    b, g, r = bgr
    hsv = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return np.array((int(hsv[0]), int(hsv[1]), int(hsv[2])))
