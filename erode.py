import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--test", default="test1.jpg")
parser.add_argument("--mask", default="test1_OUT.jpg")
parser.add_argument("--save", default="test1_eroded.jpg")
args = parser.parse_args()

img1 = cv2.imread(args.test, 3)
img2 = cv2.imread(args.mask, 0) 
seg_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, fg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
fg = cv2.bitwise_and(img, fg_mask)

'''kernel = np.ones((2,2), np.uint8) 
img2 = cv2.erode(img2, kernel, iterations=2)

width, height, depth = img1.shape
combinedImage = cv2.merge((img1, img2))'''

cv2.imwrite(args.save, fg)