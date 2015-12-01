#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

cv2.namedWindow('original')
img = cv2.imread('./test_imgs/ryosuke_pr2.jpg')
cv2.imshow('original', img)

print(img.shape) # (高さ，幅，３)

cv2.waitKey(0)
cv2.destroyAllWindows()