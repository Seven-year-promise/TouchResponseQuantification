import torch
import torch.utils.data as data
import numpy as np
import os
import math
from PIL import Image
import cv2
import csv
import sys
sys.path.append('..')
from ImageProcessing import ImageProcessor

"""
label
0: background
1: needle
2: fish
"""

class dataset_loader(data.Dataset):

    def __init__(self, cropped_size, img_dir = "dataset/Images/", ann_path = "dataset/annotation/", transforms=None, sigma=15):
        self.im_file_path = img_dir
        self.anno_file_path = ann_path
        self.cropped_size = cropped_size

        self.im_paths = os.listdir(self.im_file_path)

        self.im_pressor = ImageProcessor()

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        im_path = self.im_paths[index]
        im_name = im_path[:-4]
        im = cv2.imread(self.im_file_path + im_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, (well_x, well_y, _) = self.im_pressor.well_detection(gray)

        x_min = int(well_x - self.cropped_size / 2)
        x_max = int(well_x + self.cropped_size / 2)
        y_min = int(well_y - self.cropped_size / 2)
        y_max = int(well_y + self.cropped_size / 2)
        img = gray[y_min:y_max, x_min:x_max]

        anno_path = self.anno_file_path + im_name + "_label.tif"
        anno_im = cv2.imread(anno_path)
        anno_im = anno_im[y_min:y_max, x_min:x_max, :]
        #anno_im[np.where(anno_im == 2)] = 255
        #cv2.imshow("tif", anno_im)
        #cv2.waitKey(0)

        heatmaps = np.zeros((y_max-y_min, x_max - x_min, 2), dtype=np.double)
        heatmaps[:, :, 0] = np.array((anno_im == 1), dtype=np.double)[:, :, 0]
        heatmaps[:, :, 1] = np.array((anno_im == 2), dtype=np.double)[:, :, 0]

        """
        heatmap_visual = np.array(heatmaps[:, :, 0], np.uint8) * 255
        cv2.imshow("heatmap", heatmap_visual)
        cv2.waitKey(0)
        heatmap_visual = np.array(heatmaps[:, :, 1], np.uint8) * 255
        cv2.imshow("heatmap", heatmap_visual)
        cv2.waitKey(0)
        """
        img = np.array(img, dtype=np.float32)
        img -= 128.0
        img /= 255.0

        #img = torch.from_numpy(img.transpose((2, 0, 1)))
        #heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1)))

        # img = self.trasforms(img)
        # heatmaps = self.trasforms(heatmaps)

        return img, heatmaps

    def __len__(self):
        return len(self.im_paths)


def _croppad(img, kpt, center, w, h):
    num = len(kpt)
    height, width, _ = img.shape
    new_img = np.empty((h, w, 3), dtype=np.float32)
    new_img.fill(128)

    # calculate offset
    offset_up = -1 * (h / 2 - center[0])
    offset_left = -1 * (w / 2 - center[1])

    for i in range(num):
        kpt[i][0] -= offset_left
        kpt[i][1] -= offset_up

    st_x = 0
    ed_x = w
    st_y = 0
    ed_y = h
    or_st_x = offset_left
    or_ed_x = offset_left + w
    or_st_y = offset_up
    or_ed_y = offset_up + h

    if offset_left < 0:
        st_x = -offset_left
        or_st_x = 0
    if offset_left + w > width:
        ed_x = width - offset_left
        or_ed_x = width
    if offset_up < 0:
        st_y = -offset_up
        or_st_y = 0
    if offset_up + h > height:
        ed_y = height - offset_up
        or_ed_y = height
    new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()

    return np.ascontiguousarray(new_img), kpt


def _get_keypoints(ann):
    kpt = np.zeros((len(ann) - 2, 3))
    for i in range(2, len(ann)):
        str = ann[i]
        [x_str, y_str, vis_str] = str.split('_')
        kpt[i - 2, 0], kpt[i - 2, 1], kpt[i - 2, 2] = int(x_str), int(y_str), int(vis_str)
    return kpt


def _generate_heatmap(img, kpt, stride, sigma):
    height, width, _ = img.shape
    heatmap = np.zeros((height / stride, width / stride, len(kpt) + 1), dtype=np.float32)  # (24 points + background)
    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    for i in range(num):
        if kpt[i][2] == -1:  # not labeled
            continue
        x = kpt[i][0]
        y = kpt[i][1]
        for h in range(height):
            for w in range(width):
                xx = start + w * stride
                yy = start + h * stride
                dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                if dis > 4.6052:
                    continue
                heatmap[h][w][i] += math.exp(-dis)
                if heatmap[h][w][i] > 1:
                    heatmap[h][w][i] = 1

    heatmap[:, :, -1] = 1.0 - np.max(heatmap[:, :, :-1], axis=2)  # for background
    return heatmap

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        dataset_loader(cropped_size=220),
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True)
    for i, (input, heatmap) in enumerate(train_loader):
        print(i)