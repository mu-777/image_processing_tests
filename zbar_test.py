#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ref
# http://qiita.com/kiyota-yoji/items/7fe134a64177ed708fdd
# http://zbar.sourceforge.net/api/annotated.html

import numpy as np
import cv2
import zbar
import PIL.Image

image_path = './test_imgs/DSC_0691.jpg'

cv_img = cv2.imread(image_path)
height, width = cv_img.shape[:2]
gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
raw = gray_img.tostring()

scanner = zbar.ImageScanner()
scanner.parse_config('enable')

image = zbar.Image(width, height, 'Y800', raw)
scanner.scan(image)

for symbol in image:
    # do something useful with results
    print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data

del (image)
