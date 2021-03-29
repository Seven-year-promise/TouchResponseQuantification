import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.feature import hog
from skimage.morphology import skeletonize
SAVE_X_MIN = 100
SAVE_X_MAX = 380
SAVE_Y_MIN = 100
SAVE_Y_MAX = 380

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_path', type=str, default = './pipeline_multi-larva-touching/WT_144617_Speed25.avi',
                   help='sum the integers (default: find the max)')
args = parser.parse_args()

if __name__ == '__main__':
    cap = cv2.VideoCapture(args.video_path)

    fame_id = 0
    success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"
    frames = []
    frames.append(frame)
    while success:
        frames.append(frame[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])

        success, frame = cap.read()


    cv2.imwrite("./key_frame/last_frame.png", frames[-1])
