import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import math
import matplotlib.patches as mpatches


from NeedleDetection import circle_detection, object_detection

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, default = './split_videos/splited_images/',
                   help='sum the integers (default: find the max)')
parser.add_argument('--gt_path', type=str, default = './robustness_points/pointsnum2.txt',
                   help='sum the integers (default: find the max)')

args = parser.parse_args()
ground_truth_centers = [[280, 200],
                        [240, 200],
                        [200, 200],
                        [200, 240],
                        [200, 280],
                        [240, 280],
                        [280, 280],
                        [280, 240]]

def ComputeIOU(boxA, boxB):
    """
    xmin, ymin, xmax, ymax
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def ComputeOR(boxA, boxB):
    """
    xmin, ymin, xmax, ymax
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea)
    # return the intersection over union value
    return iou

def plot_robustness(detected_centers, ground_truth_centers, center_errors):
    all_centers = []
    for d_centers in detected_centers:
        all_centers += d_centers
    all_centers = np.array(all_centers)
    ground_truth_centers = np.array(ground_truth_centers)

    colors = [[0.1843, 0.3098, 0.3098],
              [0.6843, 0.7098, 0.3098],
              [0.1843, 0.73098, 0.3098]]

    plt.scatter(ground_truth_centers[:, 0], ground_truth_centers[:, 1], c = colors[0], s=50, label='Input Position')
    plt.scatter(all_centers[:, 0], all_centers[:, 1], s=30, c = colors[1], marker="+", label='Output of the System')
    plt.xlim((160, 320))  # 也可写成plt.xlim(-5, 5)
    plt.ylim((160, 320))

    for error, gt_center in zip(center_errors, ground_truth_centers):
        plt.text(gt_center[0]-15, gt_center[1]+3, '( error = ' + str(round(error, 2)) + ')', fontsize=10)
    plt.text(260, 170, '( error: Euclidean Distance)', fontsize=10)
    plt.text(200, 310, '( average error = ' + str(round(np.average(center_errors), 2)) + ')', fontsize=10)
    plt.xlabel("X Axis of Image Coordinate System (pixels)", fontsize=12)
    plt.ylabel("Y Axis of Image Coordinate System (pixels)", fontsize=12)
    patch1 = mpatches.Patch(color=[0.1843, 0.3098, 0.3098],  label='Input Position')
    patch2 = mpatches.Patch(color=[0.6843, 0.7098, 0.3098],  label='Output of the System')
    patch3 = mpatches.Patch(color=[0.1843, 0.73098, 0.3098], label='51 hpf')
    # patch4 = mpatches.Patch(color=[0.1843, 0.7098, 0.6098], label='54 hpf')
    # patch5 = mpatches.Patch(color=[0.1843, 0.3098, 0.6098], label='57 hpf')
    #handles = [patch1, patch2],
    plt.legend(loc = "upper right")
    plt.show()
    '''
    thresholds = np.arange(pt_num) / 100.0
    recall_rates = []
    for t in thresholds:
        fish_recall_num = 0
        for iou in IOU_list:
            if iou>t:
                fish_recall_num += 1
        recall_rates.append(fish_recall_num / fish_num)

    plt.plot(thresholds, recall_rates)
    for a, b in zip(thresholds[30:60:20], recall_rates[30:60:20]):
        plt.text(a, b, '(' + str(a) + ', ' + str(round(b, 2)) +')', fontsize=10)
    plt.ylabel("Recall Ratio", fontsize=12)
    plt.xlabel("OR Threshold", fontsize=12)
    #plt.title('Recall Ratio of Larva Detection \nAs Threshold of IoU Varies')
    
    '''


if __name__ == '__main__':
    gt_points_path = args.gt_path
    gt_points_file = open(gt_points_path, 'r')
    gt_points = []

    for line in gt_points_file.readlines():
        point = line[:-1].split(',')
        point = [int(p) for p in point]
        gt_points.append(point)

    im_cnt = 0
    centers = []
    center_error = 0
    base_im_path = args.path
    for im_id in range(1, len(gt_points) + 1):
        i_path = base_im_path + str(im_id) + '.jpg'
        print(i_path)

        im = cv2.imread(i_path)
        im = cv2.flip(im, 1)#cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        Well_success, well_info = circle_detection(gray.copy())  # centerx, centery, radius
        # print("\t the well is within:", well_info, Well_success)

        # cv2.imshow("first_frame", frame)
        # cv2.waitKey(0)
        needle_blobs, needle_centers = object_detection(gray, well_info,
                                                        threshold=30,
                                                        dis_threshold=3,
                                                        size_threshold_High=10,
                                                        size_threshold_Low=0,
                                                        what_detected='needle',
                                                        blur=False)
        Ncenter = needle_centers[0]

        print(gt_points[im_id - 1][2], gt_points[im_id - 1][3])
        distance = math.sqrt(
            (Ncenter[1] - gt_points[im_id - 1][3]) ** 2 + (Ncenter[0] - gt_points[im_id - 1][2]) ** 2)
        center_error += distance

        centers.append([Ncenter[1], Ncenter[0]])
    print('error average:', center_error / len(gt_points))

