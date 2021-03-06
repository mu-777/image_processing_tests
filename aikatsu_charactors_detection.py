#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from functions import hsv_to_bgr, bgr_to_hsv

AIKATSU_NAMES = [
    "ICHIGO",
    "AOI",
    "RAN",
    "MIZUKI",
    "OTOME",
    "YURIKA",
    "SAKURA",
    "KAEDE"
]

# BGR
AIKATSU_ANIME_HAIRS = {
    AIKATSU_NAMES[0]: (123, 235, 255),
    AIKATSU_NAMES[1]: (174, 77, 57),
    AIKATSU_NAMES[2]: (57, 63, 140),
    AIKATSU_NAMES[3]: (212, 112, 194),
    AIKATSU_NAMES[4]: (43, 138, 218),
    AIKATSU_NAMES[5]: (215, 236, 233),
    AIKATSU_NAMES[6]: (192, 169, 254),
    AIKATSU_NAMES[7]: (62, 44, 197)
}

AIKATSU_MODEL_HAIRS = {
    AIKATSU_NAMES[0]: (145, 228, 228),
    AIKATSU_NAMES[1]: (174, 88, 80),
    AIKATSU_NAMES[2]: (75, 76, 134),
    AIKATSU_NAMES[3]: (211, 125, 197),
    AIKATSU_NAMES[4]: (59, 156, 228),
    AIKATSU_NAMES[5]: (243, 255, 255),
    AIKATSU_NAMES[6]: (207, 184, 238),
    AIKATSU_NAMES[7]: (78, 60, 203)
}


def detect_h_diff(ref_hair_color_map):
    def func(hsv):
        min_name = ''
        min_hsv = (0, 0, 0)
        min_h = 100000
        for name, hair_bgr in ref_hair_color_map.items():
            hair_hsv = bgr_to_hsv(hair_bgr)
            diff = abs(hair_hsv[0] - hsv[0])
            if diff < min_h:
                min_h = diff
                min_hsv = hair_hsv
                min_name = name
        return min_name, min_hsv

    return func


def detect_bgr_diff(ref_hair_color_map):
    def func(hsv):
        bgr = hsv_to_bgr(hsv)
        min_name = ''
        min_bgr = (0, 0, 0)
        min_diff = 100000
        for name, hair_bgr in ref_hair_color_map.items():
            diff = np.sqrt(sum([(hair_bgr[0] - bgr[0]) ** 2,
                                (hair_bgr[1] - bgr[1]) ** 2,
                                (hair_bgr[2] - bgr[2]) ** 2]))
            if diff < min_diff:
                min_diff = diff
                min_bgr = hair_bgr
                min_name = name
        return min_name, bgr_to_hsv(min_bgr)

    return func


def detect_hsv_diff(ref_hair_color_map):
    def func(hsv):
        min_name = ''
        min_hsv = (0, 0, 0)
        min_diff = 100000
        for name, hair_bgr in ref_hair_color_map.items():
            hair_hsv = bgr_to_hsv(hair_bgr)
            diff = np.sqrt(sum([(hair_hsv[0] - hsv[0]) ** 2,
                                (hair_hsv[1] - hsv[1]) ** 2,
                                (hair_hsv[2] - hsv[2]) ** 2]))
            if diff < min_diff:
                min_diff = diff
                min_hsv = hair_hsv
                min_name = name
        return min_name, min_hsv

    return func


detect_aikatsu_charactors = {
    'anime_based': {'h_diff': detect_h_diff(AIKATSU_ANIME_HAIRS),
                    'bgr_diff': detect_bgr_diff(AIKATSU_ANIME_HAIRS),
                    'hsv_diff': detect_hsv_diff(AIKATSU_ANIME_HAIRS)},
    'model_based': {'h_diff': detect_h_diff(AIKATSU_MODEL_HAIRS),
                    'bgr_diff': detect_bgr_diff(AIKATSU_MODEL_HAIRS),
                    'hsv_diff': detect_hsv_diff(AIKATSU_MODEL_HAIRS)}
}



# --------------------------------------------
if __name__ == '__main__':

    import cv2
    import numpy as np
    from functions import check_img, is_skin, get_hair_color_hsv, hsv_to_bgr, bgr_to_hsv

    CASCADE_PATH = "./cascade/lbpcascade_animeface.xml"

    # IN_IMG_PATHS = ["./test_imgs/face_detecting" + str(i + 1) + ".png" for i in range(14)]
    # IN_IMG_PATHS = ["./test_imgs/hirari-hitori-kirari/face_detecting5.png"]
    IN_IMG_PATHS = ["./test_imgs/hirari-hitori-kirari/face_detecting" + str(i + 1) + ".png" for i in range(5)]
    OVERLAY_IMG_PATH = "./test_imgs/face_up3.jpg"
    OUT_IMG_PATH = None  # "./test_imgs/faces/"
    CHECK_IMG_FLAG = False


    def main(in_img_path):
        rgb_img = cv2.imread(in_img_path)
        overlay_img = cv2.imread(OVERLAY_IMG_PATH, -1)
        cascade = cv2.CascadeClassifier(CASCADE_PATH)

        faces = cascade.detectMultiScale(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY),
                                         scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
        if len(faces) <= 0:
            check_img(rgb_img, 'total', True, OUT_IMG_PATH)
            return

        for (x, y, w, h) in faces:
            face_img = rgb_img[y:y + h, x:x + w]
            if not is_skin(face_img):
                continue

            # color = hsv_to_bgr(get_hair_color_hsv(face_img))
            name, hsv = detect_aikatsu_charactors['anime_based']['bgr_diff'](get_hair_color_hsv(face_img))
            color = hsv_to_bgr(hsv)
            cv2.rectangle(rgb_img, (x, y), (x + w, y + h), color, thickness=7)

        check_img(rgb_img, 'total', True, OUT_IMG_PATH)


    for in_img_path in IN_IMG_PATHS:
        main(in_img_path)
