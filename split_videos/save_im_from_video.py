import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.feature import hog
from skimage.morphology import skeletonize

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_path', type=str, default = './3_26used.avi',
                   help='sum the integers (default: find the max)')
parser.add_argument('--save_path', type=str, default = './splited_images/',
                   help='sum the integers (default: find the max)')
args = parser.parse_args()

if __name__ == '__main__':
    cap = cv2.VideoCapture(args.video_path)
    fame_id = 0
    success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"

    while success:
        cv2.imwrite(args.save_path + str(fame_id) + '.jpg', frame)
        success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"
        fame_id += 1