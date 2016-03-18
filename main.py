#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/


import cv2
from functions import check_img, is_front_face, get_alphachannel, fill_void, is_skin
from frame_manager import FacesManager

CASCADE_PATH = "./cascade/lbpcascade_animeface.xml"
# IN_VIDEO_PATH = "./test_imgs/nanohaAs_promotion_video.mp4"
# IN_VIDEO_PATH = "./test_imgs/hirari-hitori-kirari.mp4"
IN_VIDEO_PATH = "./test_imgs/aikatsu_calendargirl_edited.mp4"
OUT_VIDEO_PATH = "./test_imgs/output.avi"
OVERLAY_IMG_PATH = "./test_imgs/face_up3.jpg"
FRAME_SIZE = (1920, 1080)
FPS = 30.0
CHECK_IMG_FLAG = False


def overlay(faces, rgb_img, overlay_img, cascade):
    if len(faces) <= 0:
        return rgb_img

    for (x, y, w, h) in faces:
        face_img = rgb_img[y:y + h, x:x + w]
        # if w < 40:
        #     continue

        if not is_skin(face_img):
            continue

        alpha_channel = fill_void(get_alphachannel(face_img))
        resized_overlay_img = cv2.resize(overlay_img, tuple((w, h)))
        mask_img = cv2.bitwise_and(resized_overlay_img, resized_overlay_img,
                                   mask=alpha_channel)
        for i, j in [(i, j) for i in range(h) for j in range(w)]:
            if any(mask_img[i, j]):
                rgb_img[y + i, x + j] = mask_img[i, j]

    return rgb_img


# --------------------------------------------
if __name__ == '__main__':

    overlay_img = cv2.imread(OVERLAY_IMG_PATH, -1)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(IN_VIDEO_PATH)
    out = cv2.VideoWriter(filename=OUT_VIDEO_PATH, fourcc=0,
                          fps=FPS, frameSize=FRAME_SIZE)
    frame_idx = 0
    faces_mgr = FacesManager()

    # フレームごとの処理
    while cap.isOpened():
        frame_idx += 1
        if frame_idx % 50 == 0:
            print("frame : %d" % frame_idx)

        ret, frame = cap.read()
        if ret is False:
            print('false')
            out.write(frame)
            continue
        faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                         scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
        faces = faces_mgr.append(faces).get_faces()
        overlayed_frame = overlay(faces, frame, overlay_img, cascade)
        out.write(overlayed_frame)

    cap.release()
    cv2.destroyAllWindows()
    out.release()
