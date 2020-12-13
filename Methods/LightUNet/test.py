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
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
        print(img.size())

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

    def predict(self, threshold):
        pred = self.model(self.input_var)
        heat = F.sigmoid(pred)
        out_needle = np.zeros(self.ori_im_size, np.uint8)
        out_fish = np.zeros(self.ori_im_size, np.uint8)

        heatmap_visual = heat[0, 0, :, :].cpu().data.numpy()
        needle_binary = np.zeros(heatmap_visual.shape, np.uint8)
        needle_binary[np.where(heatmap_visual>threshold)] = 1
        needle_binary = self.blob_tune(needle_binary)
        out_needle[self.y_min:self.y_max, self.x_min:self.x_max] = needle_binary

        #print(needle_binary, needle_binary.shape)
        #cv2.imshow("needle", needle_binary)
        #cv2.waitKey(0)

        heatmap_visual = heat[0, 1, :, :].cpu().data.numpy()
        fish_binary = np.zeros(heatmap_visual.shape, np.uint8)
        fish_binary[np.where(heatmap_visual > threshold)] = 1
        fish_binary = self.blob_tune(fish_binary)
        out_fish[self.y_min:self.y_max, self.x_min:self.x_max] = fish_binary

        #print(fish_binary, fish_binary.shape)
        #cv2.imshow("fish", out_binary*127)
        #cv2.waitKey(0)
        return out_needle, out_fish

    def get_keypoint(self, threshold):
        out_needle, out_fish = self.predict(threshold=threshold)



if __name__ == '__main__':
    time_cnt = time.time()

    unet_test = UNetTest(n_class=2, cropped_size=240, model_path="5000.pth.tar")
    unet_test.load_model()
    im = cv2.imread("dataset/Images/0.jpg")
    anno_im = cv2.imread("dataset/annotation/0_label.tif")
    unet_test.load_im(im)
    unet_test.predict()
    time_used = time.time() - time_cnt
    print("used time", time_used)
