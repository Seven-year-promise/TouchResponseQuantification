import os, sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
from Methods.LightUNet.UNet import UNet
import argparse
import torchvision.transforms as transforms
import time
from collections import defaultdict
import torch.nn.functional as F
from Methods.LightUNet.loss import dice_loss
from Methods.ImageProcessing import well_detection
from Methods.FeatureExtraction import select_blob

import numpy as np
from skimage.morphology import skeletonize
import math


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='dataset/',
                    help='enter the path for training')
parser.add_argument('--test_path', type=str, default='data//random_2816//samples_for_test.csv',
                    help='enter the path for testing')
parser.add_argument('--eval_path', type=str, default='data//random_2816//samples_for_evaluation.csv',
                    help='enter the path for evaluating')
parser.add_argument('--model_path', type=str, default='models//Liebherr10000checkpoint.pth',
                    help='enter the path for trained model')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='enter the path for training')
parser.add_argument('--batch_size', type=int, default=12,
                    help='enter the batch size for training')
parser.add_argument('--workers', type=int, default=6,
                    help='enter the number of workers for training')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='enter the weight_decay for training')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='enter the momentum for training')
parser.add_argument('--display', type=int, default=2,
                    help='enter the display for training')
parser.add_argument('--max_iter', type=int, default=160000,
                    help='enter the max iterations for training')
parser.add_argument('--test_interval', type=int, default=50,
                    help='enter the test_interval for training')
parser.add_argument('--topk', type=int, default=3,
                    help='enter the topk for training')
parser.add_argument('--start_iters', type=int, default=0,
                    help='enter the start_iters for training')
parser.add_argument('--best_model', type=float, default=12345678.9,
                    help='enter the best_model for training')
parser.add_argument('--lr_policy', type=str, default='multistep',
                    help='enter the lr_policy for training')
parser.add_argument('--policy_parameter', type=dict, default={"stepvalue":[50000, 100000, 120000], "gamma": 0.33},
                    help='enter the policy_parameter for training')
parser.add_argument('--epoch', type=int, default=400,
                    help='enter the path for training')
parser.add_argument('--lamda', type=float, default=0.0,
                    help='enter the path for training')
parser.add_argument('--save_path', type=str, default='models/',
                    help='enter the path for training')

device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu")

class UNetTest:
    def __init__(self, n_class, cropped_size, model_path):
        self.model = UNet(n_class = n_class).double()
        self.model.to(device)
        self.model.eval()
        self.cropped_size = cropped_size
        self.model_path = model_path
        self.trans = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
        ])
        self.input_var = None

        self.ori_im_size = []

    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(self.model)

    def load_im(self, im):
        # ---------------- read info -----------------------
        self.ori_im = im
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.ori_im_gray = gray
        _, (well_x, well_y, _), im_well = well_detection(im, gray)

        self.ori_im_size = gray.shape

        self.x_min = int(well_x - self.cropped_size / 2)
        self.x_max = int(well_x + self.cropped_size / 2)
        self.y_min = int(well_y - self.cropped_size / 2)
        self.y_max = int(well_y + self.cropped_size / 2)
        im_block = im_well[self.y_min:self.y_max, self.x_min:self.x_max, :]
        #cv2.imshow("needle", im_block)
        #cv2.waitKey(0)
        img = torch.from_numpy(im_block.transpose((2, 0, 1))).double() / 255
        img = self.trans(img)
        img.unsqueeze_(dim=0)
        #print(img.size())

        self.input_var = torch.autograd.Variable(img).to(device)

    def blob_tune(self, binary):
        binary = binary * 255
        ret, labels = cv2.connectedComponents(binary)
        blobs_raw = []
        for label in range(1, ret):
            coordinate = np.asarray(np.where(labels == label)).transpose()
            blobs_raw.append(coordinate)
        erosion = cv2.erode(binary, (3, 3), iterations=4)
        ret, labels = cv2.connectedComponents(erosion)
        blobs_tuned = []
        for label in range(1, ret):
            coordinate = np.asarray(np.where(labels == label)).transpose()
            if coordinate.shape[0] > 10:
                blobs_tuned.append(coordinate)
        final_blobs = select_blob(blobs_raw, blobs_tuned)
        tuned_binary = np.zeros(erosion.shape, np.uint8)
        for fblob in final_blobs:
            tuned_binary[fblob[:, 0], fblob[:, 1]] = 1

        return tuned_binary

    def select_big_blobs(self, binary, size = 20):
        binary = binary * 255
        ret, labels = cv2.connectedComponents(binary)
        blobs_tuned = []
        for label in range(1, ret):
            coordinate = np.asarray(np.where(labels == label)).transpose()
            if coordinate.shape[0] > size:
                blobs_tuned.append(coordinate)
        tuned_binary = np.zeros(binary.shape, np.uint8)
        for fblob in blobs_tuned:
            tuned_binary[fblob[:, 0], fblob[:, 1]] = 1

        return tuned_binary, blobs_tuned

    def predict(self, threshold, size):
        pred = self.model(self.input_var)
        heat = F.sigmoid(pred)
        out_needle = np.zeros(self.ori_im_size, np.uint8)
        out_fish = np.zeros(self.ori_im_size, np.uint8)

        heatmap_visual = heat[0, 0, :, :].cpu().data.numpy()
        needle_binary = np.zeros(heatmap_visual.shape, np.uint8)
        needle_binary[np.where(heatmap_visual>threshold)] = 1
        #needle_binary = self.blob_tune(needle_binary)
        out_needle[self.y_min:self.y_max, self.x_min:self.x_max] = needle_binary

        #print(needle_binary, needle_binary.shape)
        #cv2.imshow("needle", needle_binary)
        #cv2.waitKey(0)

        heatmap_visual = heat[0, 1, :, :].cpu().data.numpy()
        fish_binary = np.zeros(heatmap_visual.shape, np.uint8)
        fish_binary[np.where(heatmap_visual > threshold)] = 1
        out_fish[self.y_min:self.y_max, self.x_min:self.x_max] = fish_binary
        fish_binary, fish_blobs = self.select_big_blobs(out_fish, size=size)
        #print(fish_binary, fish_binary.shape)
        #cv2.imshow("fish", out_binary*127)
        #cv2.waitKey(0)
        return out_needle, out_fish, fish_blobs

    def find_needle_point(self, needle_mask):
        """
        :param needle_mask: the binary of the needle heat map: 0/1
        :return: the center of needle point: y, x or (h, w)
        """
        masked = needle_mask*255
        masked_inv = cv2.bitwise_not(masked)
        gray_masked = cv2.bitwise_and(self.ori_im_gray, self.ori_im_gray, mask = masked)
        gray_masked = gray_masked + masked_inv

        threshold = np.min(np.array(gray_masked, dtype=np.int))

        min_index = np.where(gray_masked == threshold)
        cy = (int)(np.round(np.average(min_index[0])))
        cx = (int)(np.round(np.average(min_index[1])))

        #needle_labelled = cv2.circle(self.ori_im, (cx, cy), 3, 255, -1)
        #cv2.imshow("needle", needle_labelled)
        #cv2.waitKey(0)

        return (cy, cx)

    def find_fish_points(self, fish_mask, needle_center, fish_blobs, percentages = []):
        """
        :param fish_mask: the binary of the fish: 0/1
        :param needle_center: the center of the needle: y, x
        :param fish_blobs: the coordinates of the area of the fish
        :param percentages: list of the points to be touched in percentage coordinate system
        :return: list of the coordinates to be touched for the closest fish to the needle
        """
        n_y, n_x = needle_center
        im_skele = self.ori_im.copy()
        distances = []
        fish_centers = []
        for blob in fish_blobs:
            c_y, c_x = np.round(np.average(np.array(blob), axis=0))
            fish_centers.append([c_y, c_x])
            #im_skele = cv2.circle(im_skele, (int(c_x), int(c_y)), 3, 255, -1)
            distances.append((n_x-c_x)**2 + (n_y-c_y)**2)
        closest_ind = np.argmin(np.array(distances))
        closest_blob = fish_blobs[closest_ind]
        closest_center = fish_centers[closest_ind]
        fish_binary = np.zeros(fish_mask.shape, dtype=np.uint8)
        fish_binary[closest_blob[:, 0], closest_blob[:, 1]] = 1

        #moments = cv2.moments(fish_binary*255)
        #cen_x = moments["m10"] / moments["m00"]
        #cen_y = moments["m01"] / moments["m00"]
        #a = moments["m20"] - moments["m00"]*cen_x*cen_x
        #b = 2*moments["m11"] - moments["m00"] * (cen_x**2 + cen_y**2);
        #c = moments["m02"] - moments["m00"]*cen_y*cen_y
        #theta = 0.5 *math.atan((2*moments["m11"]) / (moments["m20"] - moments["m02"]))
        #theta = (theta/math.pi) *180#0 if a==c else math.atan2(b, a-c)/2.0

        skeleton = skeletonize(fish_binary)
        skeleton_cor = np.where(skeleton>0)
        skeleton_cor = np.array([skeleton_cor[0], skeleton_cor[1]]).reshape(2, -1)
        point1 = skeleton_cor[:, 0]
        point2 = skeleton_cor[:, -1]
        #theta = get_angle(point1, point2, closest_center)
        fish_points = get_points(point1, point2, closest_center, percentages)
        #im_skele[np.where(skeleton==1)] = [0,255,0]
        colors = [[0, 0, 255],
                  [0, 255, 0],
                  [255, 0, 0]]
        for f_p, c in zip(fish_points, colors):
            im_skele = cv2.circle(img = im_skele, center = (f_p[1], f_p[0]), radius =1, color = c, thickness = 1)
        cv2.imshow("fish", im_skele)
        cv2.waitKey(0)
        return None

    def get_keypoint(self, threshold, size_fish):
        out_needle, out_fish, fish_blobs = self.predict(threshold=threshold, size = size_fish)
        needle_y, needle_x = self.find_needle_point(needle_mask = out_needle)
        self.find_fish_points(fish_mask=out_fish, needle_center=(needle_y, needle_x),
                              fish_blobs=fish_blobs, percentages=[0.05, 0.3, 0.7])
        #cv2.imshow("needle", out_needle*255)
        #cv2.imshow("fish", out_fish * 255)
        #cv2.waitKey(0)

        return out_needle, out_fish

def get_angle(point1, point2, fish_center):
    y1, x1 = point1
    y2, x2 = point2
    f_y, f_x = fish_center
    distance1 = (f_x - x1)**2 + (f_y - y1)**2
    distance2 = (f_x - x2)**2 + (f_y - y2)**2

    delta_x = x2-x1
    delta_y = y2-y1


    if distance1 < distance2:
        if delta_x < 0:
            theta = math.atan(delta_y / delta_x)
            theta = - (theta / math.pi) * 180
        elif delta_x > 0:
            theta = math.atan(delta_y / delta_x)
            theta = 180 - (theta / math.pi) * 180
        else:
            theta = 90
    else:
        if delta_x > 0:
            theta = math.atan(delta_y / delta_x)
            theta = - (theta / math.pi) * 180
        elif delta_x < 0:
            theta = math.atan(delta_y / delta_x)
            theta = -180 - (theta / math.pi) * 180
        else:
            theta = -90
    #print(point1, point2, fish_center, theta)
    return theta

def get_points(point1, point2, fish_center, percentages):
    y1, x1 = point1
    y2, x2 = point2
    f_y, f_x = fish_center
    distance1 = (f_x - x1)**2 + (f_y - y1)**2
    distance2 = (f_x - x2)**2 + (f_y - y2)**2

    if distance1 < distance2:
        top_head = point1
        tail_end = point2
    else:
        top_head = point2
        tail_end = point1
    percent_points = []
    for p in percentages:
        p_y = int(np.round((1-p) * top_head[0] + p * tail_end[0]))
        p_x = int(np.round((1-p) * top_head[1] + p * tail_end[1]))
        percent_points.append([p_y, p_x])
    #print(point1, point2, fish_center, theta)
    return percent_points



if __name__ == '__main__':
    time_cnt = time.time()

    unet_test = UNetTest(n_class=2, cropped_size=240, model_path="6000.pth.tar")
    unet_test.load_model()
    im = cv2.imread("dataset/test/Images/100.jpg")
    anno_im = cv2.imread("dataset/test/annotation/100_label.tif")
    unet_test.load_im(im)
    unet_test.get_keypoint(threshold=0.9)
    time_used = time.time() - time_cnt
    print("used time", time_used)
