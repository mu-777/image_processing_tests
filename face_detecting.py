#!/usr/bin/env python
# -*- coding: utf-8 -*-

# アニメ顔分類器
# https://github.com/nagadomi/lbpcascade_animeface
# 動画で検出サンプル
# http://www.takunoko.com/blog/python%E3%81%A7%E9%81%8A%E3%82%93%E3%81%A7%E3%81%BF%E3%82%8B-part1-opencv%E3%81%A7%E9%A1%94%E8%AA%8D%E8%AD%98/


import cv2
import time

cascade_path = "./cascade/lbpcascade_animeface.xml"

in_video_path = "./test_imgs/nanohaAs_promotion_video.mp4"
out_video_path = "./test_imgs/output.m4v"

# カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

# 動画のエンコード　とりあえず、これで動く拡張子はm4vで
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
# 動画ファイル読み込み
cap = cv2.VideoCapture(in_video_path)

out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (640, 360))

frame_num = 0
color = (0, 187, 254)

# フレームごとの処理
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    print("frame : %d" % frame_num)
    if len(faces) > 0:
        # 検出した顔を囲む矩形の作成
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, thickness=7)
    out.write(frame)
    frame_num += 1

cap.release()
cv2.destroyAllWindows()
out.release()
