#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/


import cv2
from functions import check_img, is_front_face, get_alphachannel, fill_void, is_skin, get_hair_color, hsv_to_bgr

CASCADE_PATH = "./cascade/lbpcascade_animeface.xml"

# IN_IMG_PATHS = ["./test_imgs/face_detecting" + str(i + 1) + ".png" for i in range(9)]
IN_IMG_PATHS = ["./test_imgs/hirari-hitori-kirari/face_detecting5.png"]
OVERLAY_IMG_PATH = "./test_imgs/face_up3.jpg"
OUT_IMG_PATH = "./test_imgs/faces/"
CHECK_IMG_FLAG = False


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

        check_img(face_img, 'face_img', CHECK_IMG_FLAG, OUT_IMG_PATH)
        if not is_skin(face_img):
            continue

        color = hsv_to_bgr(get_hair_color(face_img))
        # color = get_hair_color(face_img, is_hsv=False)
        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), color, thickness=7)

        # is_front_face(face_img)
        #
        # alpha_channel = get_alphachannel(face_img)
        # check_img(alpha_channel, 'alpha_channel', CHECK_IMG_FLAG, OUT_IMG_PATH)
        # alpha_channel = fill_void(alpha_channel)
        #
        # resized_overlay_img = cv2.resize(overlay_img, tuple((w, h)))
        # mask_img = cv2.bitwise_and(resized_overlay_img, resized_overlay_img,
        #                            mask=alpha_channel)
        # check_img(mask_img, 'mask_img', CHECK_IMG_FLAG, OUT_IMG_PATH)
        # for i, j in [(i, j) for i in range(h) for j in range(w)]:
        #     if any(mask_img[i, j]):
        #         rgb_img[y + i, x + j] = mask_img[i, j]

    check_img(rgb_img, 'total', True, OUT_IMG_PATH)


# --------------------------------------------
if __name__ == '__main__':
    for in_img_path in IN_IMG_PATHS:
        main(in_img_path)
