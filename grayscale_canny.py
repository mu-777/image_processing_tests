#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://derivecv.tumblr.com/post/63641006698

import cv2
import numpy as np
import sys

cv2.namedWindow('original')
cv2.namedWindow('edge')

img = cv2.imread('./test_imgs/ryosuke_pr2.jpg')
cv2.imshow('original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thrs1 =1000
thrs2 =10
edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
cv2.imshow('edge', edge)

cv2.waitKey(0)
cv2.destroyAllWindows()

