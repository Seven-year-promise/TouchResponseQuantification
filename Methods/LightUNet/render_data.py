import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import random
import numpy as np
import os
import math
from PIL import Image
import cv2
from PIL import Image
import csv
from util import well_detection

"""
label
0: background
1: needle
2: fish
"""
cropped_size = 240
ori_im_path = "dataset/train/Images/"
ori_anno_path = "dataset/train/annotation/"
render_im_path = "dataset/render_train/Images/"
render_anno_path = "dataset/render_train/annotation/"

if __name__ == "__main__":

    ims_name = os.listdir(ori_im_path)
    annos_name = os.listdir(ori_anno_path)
    im_anno_list = []
    for im_name in ims_name:
        name = im_name[:-4]
        im = cv2.imread(ori_im_path + im_name)
        anno = cv2.imread(ori_anno_path + name + "_label.tif")
        # anno = cv2.erode(anno, (3, 3), iterations=2)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, (well_x, well_y, _), im_well = well_detection(im, gray)

        x_min = int(well_x - cropped_size / 2)
        x_max = int(well_x + cropped_size / 2)
        y_min = int(well_y - cropped_size / 2)
        y_max = int(well_y + cropped_size / 2)
        im_block = im_well[y_min:y_max, x_min:x_max, :]
        # cv2.imshow("im", im_block)
        # cv2.waitKey(0)

        anno_im = anno[y_min:y_max, x_min:x_max, :]
        print(render_im_path + im_name)
        cv2.imwrite(render_im_path + im_name, im_block)
        cv2.imwrite(render_anno_path + name + "_label.tif", anno_im)
        # cv2.waitKey(0)
