import os
import torch
import argparse
import numpy as np
from erode import erode
from test import segment
from color_diff import contour
from keypoint_detection import mark_keypoints

parser = argparse.ArgumentParser()
#parser.add_argument('--segment_model', '-m', default='MODEL.pth', metavar='FILE', help="Specify the file in which is stored the model (default : 'MODEL.pth')")
parser.add_argument('--keypoint_model', '-m', default='./utils/pose_iter_102000.caffemodel', metavar='FILE', help="Specify the file in which is stored the keypoint model is stored")
parser.add_argument('--cpu', '-c', action='store_true', help="Do not use the cuda version of the net", default=False)
args = parser.parse_args()

if(not os.path.isfile((args.keypoint_model))):
    os.system("source ./download_keypoint.sh")
#if(not(os.path.isfile(args.segment_model))):
#    os.system("source ./download_model.sh")

def work(inputpath, outputpath):
    segmentpath = inputpath.split(".")[:-1][0] + "_seg." + inputpath.split(".")[-1]
    erodepath = inputpath.split(".")[:-1][0] + "_erod." + inputpath.split(".")[-1]
    #segment(args.model, inputpath, segmentpath, args.cpu, False, False)
    contour(inputpath, segmentpath)
    erode(inputpath, segmentpath, erodepath)
    return mark_keypoints(erodepath, outputpath, segmentpath)


if __name__ == '__main__':
    work("hand2.jpg", "hand2_out.jpg")