import os
import torch
import numpy as np
from erode import erode
from test import segment
from keypoint_detection import mark_keypoints

parser = argparse.ArgumentParser()
parser.add_argument('--segment_model', '-m', default='MODEL.pth', metavar='FILE', help="Specify the file in which is stored the model (default : 'MODEL.pth')")
parser.add_argument('--keypoint_model', '-m', default='./utils/pose_iter_102000.caffemodel', metavar='FILE', help="Specify the file in which is stored the keypoint model is stored")
parser.add_argument('--cpu', '-c', action='store_true', help="Do not use the cuda version of the net", default=False)
args = parser.parse_args()

if(not os.path.isfile(("./utils/pose_iter_102000.caffemodel"))):
    os.system("source ./download_keypoint.sh")
if(not(os.path.isfile(args.segment_model))):
    os.system("source ./download_model.sh")

def work(inputpath, outputpath):
    segmentpath = inputpath.split(".")[:-1] + "_seg." + inputpath.split(".")[-1]
    erodepath = inputpath.split(".")[:-1] + "_erod." + inputpath.split(".")[-1]
    segment(args.model, inputpath, segmentpath, args.cpu, False, False)
    erode(inputpath, segmentpath, erodepath)
    return mark_keypoints(erodepath, outputpath, segmentpath)


if __name__ == '__main__':
    work("./images.jpeg", "./output.jpeg")