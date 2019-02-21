import cv2
import numpy as np

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

def seperate(path, destination):
    frame = cv2.imread(path)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    cv2.imwrite(destination, skinMask)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

if __name__ == '__main__':
    seperate("./test1.jpg", "./test1_out.jpg")