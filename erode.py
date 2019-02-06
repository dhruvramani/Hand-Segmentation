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
kernel = np.ones((2,2), np.uint8) 
img2 = cv2.erode(img2, kernel, iterations=2)

width, height, depth = img1.shape
combinedImage = cv2.merge((img1, img2))

cv2.imwrite(args.save, combinedImage)