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
Threshold = 255

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_path', type=str, default = './pipeline_multi-larva-touching/WT_144459_Speed25.avi',
                   help='sum the integers (default: find the max)')
parser.add_argument('--save_path', type=str, default = './pipeline_multi-larva-touching/cropped/WT_150558_Speed25.avi',
                   help='sum the integers (default: find the max)')
args = parser.parse_args()

if __name__ == '__main__':
    cap = cv2.VideoCapture(args.video_path)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (SAVE_Y_MAX-SAVE_Y_MIN+1, SAVE_X_MAX-SAVE_X_MIN+1))

    fame_id = 0
    success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"
    first_frame = frame
    first_frame_grey = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    final_frame = first_frame_grey.copy()
    while success:
        new_frame = frame

        new_frame_grey = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("new_frame_grey", new_frame_grey)
        cv2.imshow("first_frame_grey", first_frame_grey)
        difference = np.abs(new_frame_grey - first_frame_grey)
        print(new_frame_grey, first_frame_grey)

        final_frame[np.where(difference > Threshold)] = np.array((final_frame[np.where(difference > Threshold)] + new_frame_grey[np.where(difference > Threshold)]) /2.0, np.uint8)
        success, frame = cap.read()

        new_frame_grey[np.where(difference < Threshold)] = 0
        difference[np.where(difference < 255)] = 0
        cv2.imshow("render_im", difference)
        cv2.waitKey(1)


    cv2.imshow("render_im", final_frame)

    cv2.waitKey(0)
