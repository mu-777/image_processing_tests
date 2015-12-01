#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/
# マスクサンプル
# http://tatabox.hatenablog.com/entry/2013/07/15/200232
# http://blanktar.jp/blog/2015/02/python-opencv-realtime-lauhgingman.html

import cv2
import time

cascade_path = "./cascade/lbpcascade_animeface.xml"

mask_img = cv2.imread("./test_imgs/face_up.jpg")
in_video_path = "./test_imgs/aikatsu_calendargirl_edited.mp4"
out_video_path = "./test_imgs/output.avi"

# カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(in_video_path)
out = cv2.VideoWriter(filename=out_video_path, fourcc=0,
                      fps=30.0, frameSize=(1920, 1080))

frame_num = 0

# フレームごとの処理
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    frame_num += 1
    print("frame : %d" % frame_num)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            resized_mask = cv2.resize(mask_img, tuple((w, h)))
            frame[y:y + h, x:x + w] = resized_mask[:, :]
    out.write(frame)
    if frame_num > 500:
        break

cap.release()
cv2.destroyAllWindows()
out.release()
