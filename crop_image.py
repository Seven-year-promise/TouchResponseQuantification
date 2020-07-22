import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.feature import hog
from skimage.morphology import skeletonize

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_path', type=str, default = './selected_images/',
                   help='sum the integers (default: find the max)')
parser.add_argument('--save_path', type=str, default = './cropped_images/',
                   help='sum the integers (default: find the max)')
args = parser.parse_args()

if __name__ == '__main__':
    im_files = os.listdir(args.image_path)
    fame_id = 0
    print(im_files)
    for im_path in im_files:
        if im_path[-3:] == 'jpg':
            print(args.image_path + im_path)
            ori_im = cv2.imread(args.image_path + im_path)
            cv2.imshow('im', ori_im[170:320, 140:290, :])
            cv2.waitKey(100)

            cv2.imwrite(args.save_path + im_path[:-4] + 'croped.jpg', ori_im[170:320, 140:290, :])