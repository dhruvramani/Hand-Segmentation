from __future__ import division
import cv2
import time
import math
import numpy as np

_PROTPATH = "./utils/pose_deploy.prototxt"
_WEIGHTPATH = "./utils/pose_iter_102000.caffemodel"
POSE_PAIRS = [  [0,1], [1,2], [2,3],
                [3,4], [0,5], [5,6],
                [6,7], [7,8], [0,9],
                [9,10], [10,11], [11,12],
                [0,13], [13,14], [14,15],
                [15,16], [0,17], [17,18],
                [18,19],[19,20] ]
npoints = 22
net = cv2.dnn.readNetFromCaffe(_PROTPATH, _WEIGHTPATH)

def mark_keypoints(path, destination, dist=True):
    frame = cv2.imread(path)
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    aspect_ratio = frameWidth / frameHeight
    inHeight = 368
    inWidth  = int(((aspect_ratio * inHeight) *8) // 8)
    inpblob  = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpblob)
    output = net.forward()

    points = []
    for i in range(npoints):
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        if prob > 0.1 :
            cv2.circle(frame, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    if(dist):
        for pair in POSE_PAIRS:
            p1, p2 = points[pair[0]], points[pair[1]]
            if p1 and p2:
                theta = (math.pi / 2) - math.atan((p2[1] - p1[1]) / (p2[0] - p1[0]))
                phi = math.pi + theta
                p3, p4 = list(p1), list(p1)
                while(frame[p3[0], p3[1]] != [0, 0, 0]):
                    p3[0] = p1[0] + 1 * math.cos(theta)
                    p3[1] = p1[1] + 1 * math.cos(theta)

                while([int(i) for i in frame[p4[0], p4[1]]] != [0, 0, 0]):
                    p4[0] = p1[0] + 1 * math.cos(phi)
                    p4[1] = p1[1] + 1 * math.cos(phi)

                cv2.line(frame, (p3[0], p3[1]), (p4[0], p4[1]), (0, 255, 0), 2)
                dist = "{0:0.1f}".format(math.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2))
                cv2.putText(frame, "{}".format(dist), (int(p1[0]) + 2, int(p1[1]) + 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    cv2.imwrite(destination, frame)



if __name__ == '__main__':
    mark_keypoints("./test_images/test5_erode.jpg", "./test_images/test5_key2.jpg")
