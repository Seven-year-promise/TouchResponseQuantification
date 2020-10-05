import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.feature import hog
from skimage.morphology import skeletonize

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_path', type=str, default = 'WT_152903_Speed25.avi',
                   help='sum the integers (default: find the max)')
parser.add_argument('--save_path', type=str, default = 'speed_changed.avi',
                   help='sum the integers (default: find the max)')
args = parser.parse_args()

if __name__ == '__main__':
    cap = cv2.VideoCapture(args.video_path)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (480, 480))

    fame_id = 0
    success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"

    while success:
        success, frame = cap.read()
        fame_id += 1
        if fame_id%30 == 0:
            #frame = cv2.flip(frame, 0)

            out.write(frame)


    cap.release()
    out.release()
