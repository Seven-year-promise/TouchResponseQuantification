import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import math

from NeedleDetection import circle_detection, object_detection

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, default = './split_videos/splited_images/',
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


    plt.scatter(ground_truth_centers[:, 0], ground_truth_centers[:, 1], s=80)
    plt.scatter(all_centers[:, 0], all_centers[:, 1], s=30, marker="+")

    for error, gt_center in zip(center_errors, ground_truth_centers):
        plt.text(gt_center[0]-5, gt_center[1]+3, '( error = ' + str(round(error, 2)) + ')', fontsize=8)

    plt.text(235, 240, '( average error = ' + str(round(np.average(center_errors), 2)) + ')', fontsize=8)
    plt.xlabel("X Axis of Image Coordinate System (pixels)", fontsize=12)
    plt.ylabel("Y Axis of Image Coordinate System (pixels)", fontsize=12)
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
    path = args.path
    detected_needle_centers = []
    center_errors = []
    for i in range(1, 9):
        print('for', i)
        base_im_path = path + str(i) + '/'
        im_files = os.listdir(base_im_path)

        im_cnt = 0
        centers = []
        center_error = 0
        for ifile in im_files:
            if ifile[-3:] != 'jpg':
                continue

            im_cnt += 1
            i_path = base_im_path + ifile
            print(i_path)

            im = cv2.imread(i_path)

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
            distance = math.sqrt((Ncenter[1]-ground_truth_centers[i-1][0])**2 + (Ncenter[0]-ground_truth_centers[i-1][1])**2)
            center_error += distance

            centers.append([Ncenter[1], Ncenter[0]])
        detected_needle_centers.append(centers)
        center_errors.append(center_error/im_cnt)
    plot_robustness(detected_needle_centers, ground_truth_centers, center_errors)
