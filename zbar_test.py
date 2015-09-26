#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import zbar
import PIL.Image

image_path = ''

scanner = zbar.ImageScanner()
# configure the reader
scanner.parse_config('enable')

pil = PIL.Image.open(sys.argv[1]).convert('L')
(width, height) = pil.size
raw = pil.tostring()

# wrap image data
image = zbar.Image(width, height, 'Y800', raw)

# scan the image for barcodes
scanner.scan(image)

# extract results
for symbol in image:
    # do something useful with results
    print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data

# clean up
del (image)