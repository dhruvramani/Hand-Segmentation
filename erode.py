import cv2
import argparse
import numpy as np

for i in range(10):
    img1 = cv2.imread("./test_images/test{}.jpg".format(i+1))
    img2 = cv2.imread("./test_images/test{}_OUT.jpg".format(i+1)) 
    seg_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    fg = cv2.bitwise_and(img1, fg_mask)
    cv2.imwrite("./test_images/test{}_erode.jpg".format(i+1), fg)