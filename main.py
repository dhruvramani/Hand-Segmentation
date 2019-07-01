import os
import torch
import argparse
import numpy as np
from sk_segment import segment
from keypoint_detection import mark_keypoints

parser = argparse.ArgumentParser()
parser.add_argument('--keypoint_model', '-m', default='./utils/pose_iter_102000.caffemodel', metavar='FILE', help="Specify the file in which is stored the keypoint model is stored")
parser.add_argument('--cpu', '-c', action='store_true', help="Do not use the cuda version of the net", default=False)
args = parser.parse_args()

if(not os.path.isfile((args.keypoint_model))):
    os.system("source ./download_keypoint.sh")

def main(inputpath, outputpath):
    segmentpath = "./hand_segment.jpg" #inputpath.split(".")[:-1][0] + "_seg." + inputpath.split(".")[-1]
    erodepath = "./hand_erod.jpg" #inputpath.split(".")[:-1][0] + "_erod." + inputpath.split(".")[-1]
    segment(inputpath, segmentpath)
    return mark_keypoints(inputpath, outputpath, segmentpath, dist=True)

if __name__ == '__main__':
    points, p_info = main("./a4.JPG", "./hand_out.jpg")
