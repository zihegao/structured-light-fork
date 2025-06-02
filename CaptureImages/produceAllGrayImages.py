# produceAllGrayImages.py
# Generates and saves composite images of horizontal Gray code and binary patterns.
# Useful for projecting patterns in structured light 3D scanning setups.

import cv2
import numpy as np
from GrayImages import GrayImage, BinaryImage

# generate Gray code image composite 
gi = GrayImage(width=600, height=32)
out_img = np.ones((320, 600+100), dtype=np.uint8)*255
startpos = 0
for i, img in gi.getIterator():
    if i.startswith('h'):
        endpos = startpos+32
        out_img[startpos:endpos, 100:] = img
        startpos = endpos

# convert to color add red lines + labels 
color_out = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
for i in range(10):
    i2 = i * 32
    color_out[i2-1:i2+2] = (255, 0, 0)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(color_out, 'Image '+str(i+1), (5, i2+22), font, 0.5, (0, 0, 0),1,cv2.LINE_AA)


cv2.imwrite("grayPatturn.png", color_out)

# repeat same fro binary code patterns
gi = BinaryImage(width=600, height=32)
out_img = np.ones((320, 600+100), dtype=np.uint8)*255
startpos = 0
for i, img in gi.getIterator():
    if i.startswith('h'):
        endpos = startpos+32
        out_img[startpos:endpos, 100:] = img
        startpos = endpos

color_out = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
for i in range(10):
    i2 = i * 32
    color_out[i2-1:i2+2] = (255, 0, 0)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(color_out, 'Image '+str(i+1), (5, i2+22), font, 0.5, (0, 0, 0),1,cv2.LINE_AA)
cv2.imwrite("binaryPatturn.png", color_out)

# show the last binary composite image
cv2.imshow("test",color_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
