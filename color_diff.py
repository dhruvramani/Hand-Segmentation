import cv2
import imutils
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

def contour(path, destination):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(image, c, -1, (255,255,255),1)
    #image = cv2.bitwise_and(image, image, mask=c)
    cv2.imwrite(destination, image)

if __name__ == '__main__':
    contour("./test_images/hand2.jpg", "./hand2_erod.jpg")