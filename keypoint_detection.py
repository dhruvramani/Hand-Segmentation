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

def net_black(frame, distrd, size=20):
    return int(np.mean(frame[distrd[0] : distrd[0] + size, :, :]))

def color_diff(color_1, color_2, threshold):
    print(color_1, color_2)
    return ((color_1[0] - color_2[0]) <= threshold) or ((color_1[1] - color_2[1]) <= threshold) or ((color_1[2] - color_2[2]) <= threshold) 

def mark_keypoints(path, destination, out_path, dist=True):
    frame = cv2.imread(path)
    outframe = cv2.imread(out_path)
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
            points.append((int(point[1]), int(point[0])))
        else :
            points.append(None)

    to_return = dict()
    if(dist):
        done = []
        allowed = [3, 6, 7, 10, 11, 14, 15, 18, 19]
        for pair in POSE_PAIRS:
            if(pair[0] not in allowed):
                continue
            done.append(pair[0])
            try :
                p1, p2 = list(points[pair[0]]), list(points[pair[1]])
            except:
                print("{} ignored".format(pair))
                continue
            if p1 and p2 and p2[0] != p1[0]:
                print(pair)
                try :
                    if(p2[1] == p1[1]):
                        p2[1] += 1
                    theta =  (math.pi / 2) + math.atan((p2[0] - p1[0]) / (p2[1] - p1[1])) 
                    p3, p4 = list(p1), list(p1)
                    dist = 0
                    inital_color = list(outframe[p3[0], p3[1]]) 
                    while(color_diff(list(outframe[p3[0], p3[1]]), inital_color) or color_diff(list(outframe[int(p1[0] + dist * math.sin(theta)), int(p1[1] + dist * math.cos(theta))]), inital_color)):
                        p3[0] = int(p1[0] + dist * math.sin(theta))
                        p3[1] = int(p1[1] + dist * math.cos(theta))
                        dist += 1
                    dist = 0
                    print(" ")
                    inital_color = list(outframe[p4[0], p4[1]])
                    while(color_diff(list(outframe[p4[0], p4[1]]), inital_color) or color_diff(list(outframe[int(p1[0] - dist * math.sin(theta)), int(p1[1] - dist * math.cos(theta))]), inital_color)):
                        p4[0] = int(p1[0] - dist * math.sin(theta))
                        p4[1] = int(p1[1] - dist * math.cos(theta))
                        dist += 1
                    print(" ")
                except :
                    print("Ignored")
                    continue
                cv2.line(outframe, (p1[1], p1[0]), (p2[1], p2[0]), (0, 0, 0), 2)
                cv2.line(outframe, (p1[1], p1[0]), (p3[1], p3[0]), (0, 255, 0), 2)
                cv2.line(outframe, (p1[1], p1[0]), (p4[1], p4[0]), (0, 255, 0), 2)
                dist = math.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
                dist = "{0:0.1f}".format(dist)
                to_return[pair[0]] = (points[pair[0]][1], points[pair[0]][1], float(dist))
                #cv2.putText(frame, dist, (int(p3[1]), int(p3[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                #cv2.putText(frame, "{}".format(dist), (int(p1[0]), int(p1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    cv2.imwrite(destination, outframe)
    return points, to_return



if __name__ == '__main__':
    mark_keypoints("./hand2.jpg", "./hand2_out.jpg", "./hand2_seg.jpg")
