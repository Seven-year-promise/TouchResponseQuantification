#!/usr/bin/python

"""
Evaluation for image segmentation.
"""

import numpy as np
import time
import os
import csv

from Methods.UNet_tf.test import *
import cv2
from Methods.FeatureExtraction import Binarization
from Methods.ImageProcessing import well_detection

import matplotlib.pyplot as plt

binarize = Binarization(method = "Binary")
otsu = Binarization(method = "Otsu")
lrb = Binarization(method = "LRB", lr_model_path="Methods/LR_models/train-on200/para700000.txt")
rg = Binarization(method = "RG")
unet_test = UNetTestTF()
#unet_test.model.load_graph_frozen(model_path="Methods/UNet_tf/ori_UNet/models-trained-on200/models_contrast_finished/UNet500.pb")

def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i


    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl
    #print(cl)

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i
    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    if n_cl_gt != 0:
        mean_IU_ = np.sum(IU) / n_cl_gt
    else:
        mean_IU_ = 0
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm, background = False):
    cl = np.unique(segm)
    if not background:
        cl = cl[1:]
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

def get_blobs(num, labels):
    blobs_raw = []
    for n in range(1, num):
        coordinate = np.where(labels == n)
        blobs_raw.append(coordinate)

    return blobs_raw

def get_iou(blobA, blobB, ori_shape):
    maskA = np.zeros(ori_shape, np.uint8)
    maskB = np.zeros(ori_shape, np.uint8)
    maskA[blobA] = 1
    maskB[blobB] = 1
    #cv2.imshow("maskA", maskA*255)
    #cv2.imshow("maskB", maskB*255)
    #cv2.waitKey(0)
    AB = np.sum(np.logical_and(maskA, maskB))
    A = np.sum(maskA)
    B = np.sum(maskB)
    #print(A, B, AB)

    return AB / (A + B - AB)

def recall_false_ratio(eval_segm, gt_segm, threshold, gt_num, gt_blobs):
    '''
    recall_ratio: TP / (TP + TN)
    false_ratio: FP / (TP + FP)
    correct_ratio: CP / (TP + FP)
    '''

    check_size(eval_segm, gt_segm)

    eval_ret, eval_labels = cv2.connectedComponents(eval_segm)
    #cv2.imshow("label", np.array(eval_labels*(255/eval_ret), np.uint8))

    eval_blobs = get_blobs(eval_ret, eval_labels)


    eval_num = len(eval_blobs)

    #print("BEGIN", gt_ret, eval_ret)
    eval_found_flag = np.zeros(eval_num, np.uint8)
    gt_found_flag = np.zeros(gt_num, np.uint8)

    for g_n in range(gt_num):
        gt_blob = gt_blobs[g_n]
        for e_n in range(eval_num):
            eval_blob = eval_blobs[e_n]
            iou = get_iou(gt_blob, eval_blob, eval_segm.shape)
            #print("iou", iou)
            if iou > threshold:
                gt_found_flag[g_n] = 1
                eval_found_flag[e_n] = 1
            #print(gt_found_flag)

    #print("END FOR ONE")
    TP = np.sum(gt_found_flag)
    TN = gt_num- TP
    FP = eval_num - np.sum(eval_found_flag)
    CP = np.sum(eval_found_flag)

    recall_ratio = TP / (gt_num)
    false_ratio = FP / (eval_num)
    correct_ratio = CP / (eval_num)
    return recall_ratio, false_ratio, correct_ratio

def recall_false_ratio_by_num(eval_segm, gt_segm, threshold, larva_num = 5):
    '''
    recall_ratio: TP / (TP + TN)
    false_ratio: FP / (TP + FP)
    correct_ratio: CP / (TP + FP)
    '''
    gt_ret, gt_labels = cv2.connectedComponents(gt_segm)
    gt_blobs = get_blobs(gt_ret, gt_labels)
    gt_num = len(gt_blobs)

    if gt_num == larva_num:
        check_size(eval_segm, gt_segm)

        eval_ret, eval_labels = cv2.connectedComponents(eval_segm)
        #cv2.imshow("label", np.array(eval_labels*(255/eval_ret), np.uint8))

        eval_blobs = get_blobs(eval_ret, eval_labels)


        eval_num = len(eval_blobs)

        #print("BEGIN", gt_ret, eval_ret)
        eval_found_flag = np.zeros(eval_num, np.uint8)
        gt_found_flag = np.zeros(gt_num, np.uint8)

        for g_n in range(gt_num):
            gt_blob = gt_blobs[g_n]
            for e_n in range(eval_num):
                eval_blob = eval_blobs[e_n]
                iou = get_iou(gt_blob, eval_blob, eval_segm.shape)
                #print("iou", iou)
                if iou > threshold:
                    gt_found_flag[g_n] = 1
                    eval_found_flag[e_n] = 1
                #print(gt_found_flag)

        #print("END FOR ONE")
        TP = np.sum(gt_found_flag)
        TN = gt_num- TP
        FP = eval_num - np.sum(eval_found_flag)
        CP = np.sum(eval_found_flag)

        recall_ratio = TP / (gt_num)
        false_ratio = FP / (eval_num)
        correct_ratio = CP / (eval_num)
        return recall_ratio, false_ratio, correct_ratio
    else:
        return None, None, None
"""
-------------------------------------
test PC JI and recall correct ratio for Binarization, Otsu, LRB, RG, U-Net
-------------------------------------
"""
def test_binarization(im_anno_list):
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        binary = binarize.Binary(im_well, needle_thr=180)

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1
        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)

        accuracy = mean_accuracy(binary, anno)
        ave_acc += accuracy
        iu = mean_IU(binary, anno)
        ave_iu += iu
    time_used = time.time() - time_cnt
    print("average accuracy", ave_acc / num)
    print("average iu", ave_iu / num)

    print("time per frame", time_used / num)

def binarization_recall_correct_ratio(im_anno_list, thresholds, save_path = "./Method/Eval_All_Method/Binarization/"):
    num = len(im_anno_list)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recall_ratio_path = save_path + "recall_ratio" + ".csv"
    correct_ratio_path = save_path + "correct_ratio" + ".csv"
    recall_ratio_csv_file = open(recall_ratio_path, "w", newline="")
    recall_ratio_csv_writer = csv.writer(recall_ratio_csv_file, delimiter=",")
    correct_ratio_csv_file = open(correct_ratio_path, "w", newline="")
    correct_ratio_csv_writer = csv.writer(correct_ratio_csv_file, delimiter=",")
    recall_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())
    correct_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())

    for i, im_anno in enumerate(im_anno_list):
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        binary = binarize.Binary(im_well, needle_thr=180)

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1

        gt_ret, gt_labels = cv2.connectedComponents(anno)
        gt_blobs = get_blobs(gt_ret, gt_labels)
        gt_num = len(gt_blobs)


        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)
        recall_ratio_list = []
        recall_ratio_list.append(i)
        correct_ratio_list = []
        correct_ratio_list.append(i)
        for t in thresholds:
            recall_ratio, _, correct_ratio = recall_false_ratio(binary, anno, t,
                                                                          gt_num, gt_blobs)
            recall_ratio_list.append(recall_ratio)
            correct_ratio_list.append(correct_ratio)
        recall_ratio_csv_writer.writerow(recall_ratio_list)
        correct_ratio_csv_writer.writerow(correct_ratio_list)

    recall_ratio_csv_file.close()
    correct_ratio_csv_file.close()
    print("binarization recall and correct ratio, finished")


def test_Otsu(im_anno_list):
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        binary = otsu.Otsu(im_well)

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1
        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)

        accuracy = mean_accuracy(binary, anno)
        ave_acc += accuracy
        iu = mean_IU(binary, anno)
        ave_iu += iu
    time_used = time.time() - time_cnt
    print("average accuracy", ave_acc / num)
    print("average iu", ave_iu / num)

    print("time per frame", time_used / num)

def Otsu_recall_correct_ratio(im_anno_list, thresholds, save_path = "./Method/Eval_All_Method/Otsu/"):
    num = len(im_anno_list)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recall_ratio_path = save_path + "recall_ratio" + ".csv"
    correct_ratio_path = save_path + "correct_ratio" + ".csv"
    recall_ratio_csv_file = open(recall_ratio_path, "w", newline="")
    recall_ratio_csv_writer = csv.writer(recall_ratio_csv_file, delimiter=",")
    correct_ratio_csv_file = open(correct_ratio_path, "w", newline="")
    correct_ratio_csv_writer = csv.writer(correct_ratio_csv_file, delimiter=",")
    recall_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())
    correct_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())

    for i, im_anno in enumerate(im_anno_list):
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

        binary = otsu.Otsu(im_well)

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1

        gt_ret, gt_labels = cv2.connectedComponents(anno)
        gt_blobs = get_blobs(gt_ret, gt_labels)
        gt_num = len(gt_blobs)


        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)
        recall_ratio_list = []
        recall_ratio_list.append(i)
        correct_ratio_list = []
        correct_ratio_list.append(i)
        for t in thresholds:
            recall_ratio, _, correct_ratio = recall_false_ratio(binary, anno, t,
                                                                          gt_num, gt_blobs)
            recall_ratio_list.append(recall_ratio)
            correct_ratio_list.append(correct_ratio)
        recall_ratio_csv_writer.writerow(recall_ratio_list)
        correct_ratio_csv_writer.writerow(correct_ratio_list)

    recall_ratio_csv_file.close()
    correct_ratio_csv_file.close()
    print("Otsu recall and correct ratio, finished")

def test_LRB(im_anno_list):
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        binary = lrb.LRB(im_well, well_infos=(well_centerx, well_centery, well_radius))

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1
        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)

        accuracy = mean_accuracy(binary, anno)
        ave_acc += accuracy
        iu = mean_IU(binary, anno)
        ave_iu += iu
    time_used = time.time() - time_cnt
    print("average accuracy", ave_acc / num)
    print("average iu", ave_iu / num)

    print("time per frame", time_used / num)

def LRB_recall_correct_ratio(im_anno_list, thresholds, save_path = "./Method/Eval_All_Method/LRB/"):
    num = len(im_anno_list)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recall_ratio_path = save_path + "recall_ratio" + ".csv"
    correct_ratio_path = save_path + "correct_ratio" + ".csv"
    recall_ratio_csv_file = open(recall_ratio_path, "w", newline="")
    recall_ratio_csv_writer = csv.writer(recall_ratio_csv_file, delimiter=",")
    correct_ratio_csv_file = open(correct_ratio_path, "w", newline="")
    correct_ratio_csv_writer = csv.writer(correct_ratio_csv_file, delimiter=",")
    recall_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())
    correct_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())

    for i, im_anno in enumerate(im_anno_list):
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        binary = lrb.LRB(im_well, well_infos=(well_centerx, well_centery, well_radius))

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1

        gt_ret, gt_labels = cv2.connectedComponents(anno)
        gt_blobs = get_blobs(gt_ret, gt_labels)
        gt_num = len(gt_blobs)


        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)
        recall_ratio_list = []
        recall_ratio_list.append(i)
        correct_ratio_list = []
        correct_ratio_list.append(i)
        for t in thresholds:
            recall_ratio, _, correct_ratio = recall_false_ratio(binary, anno, t,
                                                                          gt_num, gt_blobs)
            recall_ratio_list.append(recall_ratio)
            correct_ratio_list.append(correct_ratio)
        recall_ratio_csv_writer.writerow(recall_ratio_list)
        correct_ratio_csv_writer.writerow(correct_ratio_list)

    recall_ratio_csv_file.close()
    correct_ratio_csv_file.close()
    print("LRB recall and correct ratio, finished")

def test_RG(im_anno_list):

    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        binary = rg.RG(im_well, threshold = 5)

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1
        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)

        accuracy = mean_accuracy(binary, anno)
        ave_acc += accuracy
        iu = mean_IU(binary, anno)
        ave_iu += iu
    time_used = time.time() - time_cnt
    print("average accuracy", ave_acc / num)
    print("average iu", ave_iu / num)

    print("time per frame", time_used / num)

def RG_recall_correct_ratio(im_anno_list, thresholds, save_path = "./Method/Eval_All_Method/RG/"):
    num = len(im_anno_list)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recall_ratio_path = save_path + "recall_ratio" + ".csv"
    correct_ratio_path = save_path + "correct_ratio" + ".csv"
    recall_ratio_csv_file = open(recall_ratio_path, "w", newline="")
    recall_ratio_csv_writer = csv.writer(recall_ratio_csv_file, delimiter=",")
    correct_ratio_csv_file = open(correct_ratio_path, "w", newline="")
    correct_ratio_csv_writer = csv.writer(correct_ratio_csv_file, delimiter=",")
    recall_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())
    correct_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())

    for i, im_anno in enumerate(im_anno_list):
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        binary = rg.RG(im_well, threshold=5)

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1

        gt_ret, gt_labels = cv2.connectedComponents(anno)
        gt_blobs = get_blobs(gt_ret, gt_labels)
        gt_num = len(gt_blobs)


        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)
        recall_ratio_list = []
        recall_ratio_list.append(i)
        correct_ratio_list = []
        correct_ratio_list.append(i)
        for t in thresholds:
            recall_ratio, _, correct_ratio = recall_false_ratio(binary, anno, t,
                                                                          gt_num, gt_blobs)
            recall_ratio_list.append(recall_ratio)
            correct_ratio_list.append(correct_ratio)
        recall_ratio_csv_writer.writerow(recall_ratio_list)
        correct_ratio_csv_writer.writerow(correct_ratio_list)

    recall_ratio_csv_file.close()
    correct_ratio_csv_file.close()
    print("RG recall and correct ratio, finished")

def test_UNet(im_anno_list):

    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        unet_test.model.load_graph_frozen(
            model_path="./Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/UNet30000.pb")
        unet_test.load_im(im)
        needle_binary, fish_binary, _, _ = unet_test.predict(threshold=0.9, size=12) # size not used
        binary = np.zeros(needle_binary.shape, np.uint8)
        binary[np.where(needle_binary > 0)] = 1
        binary[np.where(fish_binary > 0)] = 1

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1
        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)

        accuracy = mean_accuracy(binary, anno)
        ave_acc += accuracy
        iu = mean_IU(binary, anno)
        ave_iu += iu
    time_used = time.time() - time_cnt
    print("average accuracy", ave_acc / num)
    print("average iu", ave_iu / num)

    print("time per frame", time_used / num)

def UNet_recall_correct_ratio(im_anno_list, thresholds, save_path = "./Method/Eval_All_Method/U-Net/"):
    unet_test.model.load_graph_frozen(
        model_path="./Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/UNet30000.pb")
    num = len(im_anno_list)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recall_ratio_path = save_path + "recall_ratio" + ".csv"
    correct_ratio_path = save_path + "correct_ratio" + ".csv"
    recall_ratio_csv_file = open(recall_ratio_path, "w", newline="")
    recall_ratio_csv_writer = csv.writer(recall_ratio_csv_file, delimiter=",")
    correct_ratio_csv_file = open(correct_ratio_path, "w", newline="")
    correct_ratio_csv_writer = csv.writer(correct_ratio_csv_file, delimiter=",")
    recall_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())
    correct_ratio_csv_writer.writerow(["threshold"] + thresholds.tolist())

    for i, im_anno in enumerate(im_anno_list):
        im, anno_needle, anno_fish = im_anno
        unet_test.load_im(im)
        needle_binary, fish_binary, _, _ = unet_test.predict(threshold=0.9, size=12) # size not used
        binary = np.zeros(needle_binary.shape, np.uint8)
        binary[np.where(needle_binary > 0)] = 1
        binary[np.where(fish_binary > 0)] = 1

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1

        gt_ret, gt_labels = cv2.connectedComponents(anno)
        gt_blobs = get_blobs(gt_ret, gt_labels)
        gt_num = len(gt_blobs)


        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)
        recall_ratio_list = []
        recall_ratio_list.append(i)
        correct_ratio_list = []
        correct_ratio_list.append(i)
        for t in thresholds:
            recall_ratio, _, correct_ratio = recall_false_ratio(binary, anno, t,
                                                                          gt_num, gt_blobs)
            recall_ratio_list.append(recall_ratio)
            correct_ratio_list.append(correct_ratio)
        recall_ratio_csv_writer.writerow(recall_ratio_list)
        correct_ratio_csv_writer.writerow(correct_ratio_list)

    recall_ratio_csv_file.close()
    correct_ratio_csv_file.close()
    print("U-Net recall and correct ratio, finished")

"""
-------------------------------------
test PC JI of U-Net: needle and larva separately
-------------------------------------
"""
def test_UNet_detailed(im_anno_list, save = True):
    ave_needle_acc = 0
    ave_fish_acc = 0
    ave_needle_iu = 0
    ave_fish_iu = 0
    num_needle = 0
    num_fish = 0
    num_im = len(im_anno_list)
    time_cnt = time.time()
    i = 0
    for im_anno in im_anno_list:
        i += 1
        im, anno_needle, anno_fish = im_anno

        unet_test.load_im(im)
        needle_binary, fish_binary, im_with_points, fish_points = unet_test.get_keypoint(threshold=0.9, size_fish=12)

        if save:
            save_im = np.zeros(needle_binary.shape, np.uint8)
            save_im[np.where(needle_binary == 1)] = 1
            save_im[np.where(fish_binary == 1)] = 2
            cv2.imwrite("GUI_saved/" + str(i) + "im_with_points.jpg", im_with_points)

        if len(np.where(anno_needle == 1)[0]) > 0:
            acc_needle = mean_accuracy(needle_binary, anno_needle)
            ave_needle_acc += acc_needle
            iu_needle = mean_IU(needle_binary, anno_needle)
            ave_needle_iu += iu_needle
            num_needle += 1

        if len(np.where(anno_fish == 1)[0]) > 0:
            acc_fish = mean_accuracy(fish_binary, anno_fish)
            ave_fish_acc += acc_fish
            iu_fish = mean_IU(fish_binary, anno_fish)
            ave_fish_iu += iu_fish
            num_fish += 1
        # cv2.imshow("binary", binary*255)
        # cv2.waitKey(0)
        # cv2.imshow("anno", anno*255)
        # cv2.waitKey(0)

    time_used = time.time() - time_cnt
    print("average needle accuracy", ave_needle_acc / num_needle)
    print("average needle iu", ave_needle_iu / num_needle)

    print("average fish accuracy", ave_fish_acc / num_fish)
    print("average fish iu", ave_fish_iu / num_fish)

    print("time per frame", time_used / num_im)

"""
-------------------------------------
test Recall correct ratio of U-Net: needle and larva separately
-------------------------------------
"""

def UNet_detailed_recall_false_ratio(im_anno_list, threshold):
    ave_needle_recall_ratio = 0
    ave_fish_recall_ratio = 0
    ave_needle_false_ratio = 0
    ave_fish_false_ratio = 0
    num_needle = 0
    num_fish = 0
    num_im = len(im_anno_list)
    time_cnt = time.time()
    i = 0
    for im_anno in im_anno_list:
        i += 1
        im, anno_needle, anno_fish = im_anno

        unet_test.load_im(im)
        needle_binary, fish_binary, im_with_points, fish_points = unet_test.get_keypoint(threshold=0.9, size_fish=44)

        if len(np.where(anno_needle == 1)[0]) > 0:
            needle_recall_ratio, needle_false_ratio = recall_false_ratio(needle_binary, anno_needle, threshold)
            ave_needle_recall_ratio += needle_recall_ratio
            ave_needle_false_ratio += needle_false_ratio

            num_needle += 1

        if len(np.where(anno_fish == 1)[0]) > 0:
            fish_recall_ratio, fish_false_ratio = recall_false_ratio(fish_binary, anno_fish, threshold)
            ave_fish_recall_ratio += fish_recall_ratio
            ave_fish_false_ratio += fish_false_ratio

            num_fish += 1
        # cv2.imshow("binary", binary*255)
        # cv2.waitKey(0)
        # cv2.imshow("anno", anno*255)
        # cv2.waitKey(0)

    return ave_needle_recall_ratio / num_needle, \
           ave_needle_false_ratio / num_needle, \
           ave_fish_recall_ratio / num_fish, \
           ave_fish_false_ratio / num_fish

"""
def UNet_larva_recall_false_ratio(im_anno_list, threshold, larva_num):
    ave_fish_recall_ratio = 0
    ave_fish_correct_ratio = 0
    num_fish = 0
    num_im = len(im_anno_list)
    time_cnt = time.time()
    i = 0
    for im_anno in im_anno_list:
        i += 1
        im, anno_needle, anno_fish = im_anno

        unet_test.load_im(im)
        needle_binary, fish_binary, im_with_points, fish_points = unet_test.get_keypoint(threshold=0.9, size_fish=12)

        if len(np.where(anno_fish == 1)[0]) > 0:
            fish_recall_ratio, _, fish_correct_ratio = recall_false_ratio(fish_binary, anno_fish, threshold, gt_numt)
            if fish_recall_ratio is not None:
                ave_fish_recall_ratio += fish_recall_ratio
                ave_fish_correct_ratio += fish_correct_ratio

                num_fish += 1
        # cv2.imshow("binary", binary*255)
        # cv2.waitKey(0)
        # cv2.imshow("anno", anno*255)
        # cv2.waitKey(0)

    return ave_fish_recall_ratio / num_fish, \
           ave_fish_correct_ratio / num_fish
"""

"""
-------------------------------------
test Recall correct ratio of U-Net: needle and larva separately and 
with different number of larvae in each image
-------------------------------------
"""
def test_Unet_larva_recall_false_ratio_by_num(im_anno_list, thre_steps = 100, save_path = "Methods/UNet_tf/ori_UNet/models-trained-on200-2/"):
    unet_test.model.load_graph_frozen(
        model_path="./Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/UNet30000.pb")

    threshold = np.arange(thre_steps)/thre_steps
    markers = [".", "s", "*", "h"]
    labels = ["2 larvae", "3 larvae", "4 larvae", "5 larvae"]
    all_recall_ratios = []
    all_correct_ratios = []
    for l_n in range(2, 6):
        recall_ratios_num = [] # for each number of larvae
        correct_ratios_num = []
        larva_recall_ratio_path = save_path + "larva_recall_ratio" + str(l_n) + ".csv"
        larva_correct_ratio_path = save_path + "larva_correct_ratio" + str(l_n) + ".csv"
        larva_recall_ratio_csv_file = open(larva_recall_ratio_path, "w", newline="")
        larva_recall_ratio_csv_writer = csv.writer(larva_recall_ratio_csv_file, delimiter=",")
        larva_correct_ratio_csv_file = open(larva_correct_ratio_path, "w", newline="")
        larva_correct_ratio_csv_writer = csv.writer(larva_correct_ratio_csv_file, delimiter=",")
        larva_recall_ratio_csv_writer.writerow(["threshold"] + threshold.tolist())
        larva_correct_ratio_csv_writer.writerow(["threshold"] + threshold.tolist())
        i = 0
        for im_anno in im_anno_list:
            im, _, anno_fish = im_anno

            gt_ret, gt_labels = cv2.connectedComponents(anno_fish)
            gt_blobs = get_blobs(gt_ret, gt_labels)
            gt_num = len(gt_blobs)
            if gt_num == l_n:
                unet_test.load_im(im)
                _, _, fish_binary, _, _ = unet_test.get_keypoint(threshold=0.9, size_fish=12)

                if len(np.where(anno_fish == 1)[0]) > 0:
                    recall_ratios_thre = []  # for each threshold with one image
                    correct_ratios_thre = []
                    for t in threshold:
                        fish_recall_ratio, _, fish_correct_ratio = recall_false_ratio(fish_binary, anno_fish, t,
                                                                                      gt_num, gt_blobs)
                        if fish_recall_ratio is not None:
                            recall_ratios_thre.append(fish_recall_ratio)
                            correct_ratios_thre.append(fish_correct_ratio)
                        else:
                            recall_ratios_thre.append(-1)
                            correct_ratios_thre.append(-1)
                    recall_ratios_num.append(recall_ratios_thre)
                    correct_ratios_num.append(correct_ratios_thre)

                    larva_recall_ratio_csv_writer.writerow([i] + recall_ratios_thre)
                    larva_correct_ratio_csv_writer.writerow([i] + correct_ratios_thre)
            i += 1
        larva_recall_ratio_csv_file.close()
        larva_correct_ratio_csv_file.close()
        all_recall_ratios.append(recall_ratios_num)
        all_recall_ratios.append(correct_ratios_num)

    """
    for (r_rs, marker, label) in zip(all_recall_ratios, markers, labels):
        plt.plot(threshold, r_rs, marker = marker, label = label)

    plt.legend(loc="best")
    plt.xlabel("Threshold of IOU")
    plt.ylabel("Recall Ratio")
    plt.title("Recall ratio with different numbers of larvae in the well")
    plt.show()

    for (f_rs, marker, label) in zip(all_false_ratios, markers, labels):
        plt.plot(threshold, f_rs, marker = marker, label = label)

    plt.legend(loc="best")
    plt.xlabel("Threshold of IOU")
    plt.ylabel("Correct Detection Ratio")
    plt.title("Correct detection ratio with different numbers of larvae in the well")
    plt.show()
    """


"""
-------------------------------------
select the suitable threshold for the size thresholding for the postprocessing of U-Net
-------------------------------------
"""
def test_UNet_select_size_thre(im_anno_list, save = False):
    unet_test.model.load_graph_frozen(model_path="./Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/UNet30000.pb")
    ave_needle_accs = []
    ave_fish_accs = []
    ave_needle_ius = []
    ave_fish_ius = []
    save_path = "./Methods/UNet_tf/ori_UNet/models-trained-on200-2/"
    PC_Larva_path = save_path + "size_thre_PC_Larva.csv"
    #if os.path.exists(PC_Larva_path) == False:
    #    os.makedirs(PC_Larva_path)
    JI_Larva_path = save_path + "size_thre_JI_Larva.csv"
    #if os.path.exists(JI_Larva_path) == False:
    #    os.makedirs(JI_Larva_path)
    PC_Larva_csv_file = open(PC_Larva_path, "w", newline="")
    PC_Larva_csv_writer = csv.writer(PC_Larva_csv_file, delimiter=",")
    JI_Larva_csv_file = open(JI_Larva_path, "w", newline="")
    JI_Larva_csv_writer = csv.writer(JI_Larva_csv_file, delimiter=",")
    for threshold in range(0, 70, 1):
        ave_needle_acc = 0
        ave_fish_acc = 0
        ave_needle_iu = 0
        ave_fish_iu = 0
        num_needle = 0
        num_fish = 0
        num_im = len(im_anno_list)
        time_cnt = time.time()
        i = 0
        for im_anno in im_anno_list:
            i += 1
            im, anno_needle, anno_fish = im_anno

            unet_test.load_im(im)
            needle_binary, _, fish_binary, im_with_points, fish_points = unet_test.get_keypoint(threshold=0.9, size_fish=threshold)

            if save:
                save_im = np.zeros(needle_binary.shape, np.uint8)
                save_im[np.where(needle_binary == 1)] = 1
                save_im[np.where(fish_binary == 1)] = 2
                cv2.imwrite("GUI_saved/" + str(i) + "im_with_points.jpg", im_with_points)

            if len(np.where(anno_needle == 1)[0]) > 0:
                acc_needle = mean_accuracy(needle_binary, anno_needle)
                ave_needle_acc += acc_needle
                iu_needle = mean_IU(needle_binary, anno_needle)
                ave_needle_iu += iu_needle
                num_needle += 1

            if len(np.where(anno_fish == 1)[0]) > 0:
                acc_fish = mean_accuracy(fish_binary, anno_fish)
                ave_fish_acc += acc_fish
                iu_fish = mean_IU(fish_binary, anno_fish)
                ave_fish_iu += iu_fish
                num_fish += 1
            # cv2.imshow("binary", binary*255)
            # cv2.waitKey(0)
            # cv2.imshow("anno", anno*255)
            # cv2.waitKey(0)
        ave_needle_acc = ave_needle_acc / num_needle
        ave_needle_iu = ave_needle_iu / num_needle
        ave_fish_acc = ave_fish_acc / num_fish
        ave_fish_iu = ave_fish_iu / num_fish

        print("threshold:", threshold)
        print("average needle accuracy", ave_needle_acc)
        print("average needle iu", ave_needle_iu)

        print("average fish accuracy", ave_fish_acc)
        print("average fish iu", ave_fish_iu)
        """
        ave_needle_accs.append(ave_needle_acc)
        ave_needle_ius.append(ave_needle_iu)
        ave_fish_accs.append(ave_fish_acc)
        ave_fish_ius.append(ave_fish_iu)
        """

        PC_Larva_csv_writer.writerow([threshold, ave_fish_acc])
        JI_Larva_csv_writer.writerow([threshold, ave_fish_iu])

    PC_Larva_csv_file.close()
    JI_Larva_csv_file.close()
    """
    plt.plot(ave_fish_accs)
    plt.plot(ave_fish_ius)
    plt.show()
    time_used = time.time() - time_cnt


    print("time per frame", time_used / num_im)
    """

"""
-------------------------------------
select the best model (training epoch) of U-net before the postprocessing of U-Net
-------------------------------------
"""

def UNet_select_epoch(im_anno_list, save_path = "Methods/UNet_tf/ori_UNet/models-trained-on200-2/"):
    """
    test all models of U-Net with different augmentation methods and find out the best model on the evaluation dataset
    :param im_anno_list:
    :param save_path:
    :return:
    """
    model_dirs = ["Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_contrast/",
                  "Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_contrast_noise/",
                  "Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_noise/",
                  "Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_ori/",
                  "Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/",
                  "Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast_noise/",
                  "Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation/",
                  "Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_noise/"]
    eval_csv_results= ["random_contrast",
                       "random_contrast_gaussian_noise",
                       "gaussian_noise",
                       "without_augmentation",
                       "random_rotation_and_contrast",
                       "random_rotation_contrast_gaussian_noise",
                       "random_rotation",
                       "random_rotation_gaussian_noise"]
    model_types = ["random contrast",
                   "random contrast and gaussian noise",
                   "gaussian noise",
                   "without augmentation",
                   "random rotation and contrast",
                   "random rotation and contrast and gaussian noise",
                   "random rotation",
                   "random rotation and gaussian noise"]

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]
    fig, axs = plt.subplots(2, 2)
    lines_axis00 = []
    lines_axis01 = []
    lines_axis10 = []
    lines_axis11 = []

    for model_dir, model_type, color, eval_file_name in zip(model_dirs[3:], model_types[3:], COLORS[3:], eval_csv_results[3:]):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pb')]
        PC_Needle_path = save_path + eval_file_name + "_PC_Needle.csv"
        JI_Needle_path = save_path + eval_file_name + "_JI_Needle.csv"
        PC_Larva_path = save_path + eval_file_name + "_PC_Larva.csv"
        JI_Larva_path = save_path + eval_file_name + "_JI_Larva.csv"
        PC_Needle_csv_file = open(PC_Needle_path, "w", newline="")
        PC_Needle_csv_writer = csv.writer(PC_Needle_csv_file, delimiter=",")
        JI_Needle_csv_file = open(JI_Needle_path, "w", newline="")
        JI_Needle_csv_writer = csv.writer(JI_Needle_csv_file, delimiter=",")
        PC_Larva_csv_file = open(PC_Larva_path, "w", newline="")
        PC_Larva_csv_writer = csv.writer(PC_Larva_csv_file, delimiter=",")
        JI_Larva_csv_file = open(JI_Larva_path, "w", newline="")
        JI_Larva_csv_writer = csv.writer(JI_Larva_csv_file, delimiter=",")

        def model_num(x):
            return (int(x[4:-3]))

        sorted_files = sorted(model_files, key=model_num)

        file_num = len(sorted_files)
        epoches = np.arange(1, file_num+1)*500

        ave_needle_accs = []
        ave_fish_accs = []
        ave_needle_ius = []
        ave_fish_ius = []
        print(sorted_files)
        file_cnt = 1
        for m_f in sorted_files:
            #print(m_f)
            unet_test.model.load_graph_frozen(model_path=model_dir + m_f)
            ave_needle_acc = 0
            ave_fish_acc = 0
            ave_needle_iu = 0
            ave_fish_iu = 0
            num_needle = 0
            num_fish = 0
            num_im = len(im_anno_list)
            i = 0
            for im_anno in im_anno_list:
                i += 1
                im, anno_needle, anno_fish = im_anno

                unet_test.load_im(im)
                needle_binary, fish_binary, _, im_with_points, fish_points = unet_test.get_keypoint(threshold=0.9,
                                                                                                 size_fish=12)

                if len(np.where(anno_needle == 1)[0]) > 0:
                    acc_needle = mean_accuracy(needle_binary, anno_needle)
                    ave_needle_acc += acc_needle
                    iu_needle = mean_IU(needle_binary, anno_needle)
                    ave_needle_iu += iu_needle
                    num_needle += 1

                if len(np.where(anno_fish == 1)[0]) > 0:
                    acc_fish = mean_accuracy(fish_binary, anno_fish)
                    ave_fish_acc += acc_fish
                    iu_fish = mean_IU(fish_binary, anno_fish)
                    ave_fish_iu += iu_fish
                    num_fish += 1
                # cv2.imshow("binary", binary*255)
                # cv2.waitKey(0)
                # cv2.imshow("anno", anno*255)
                # cv2.waitKey(0)
            ave_needle_acc /= num_needle
            ave_needle_iu /= num_needle
            ave_fish_acc /= num_fish
            ave_fish_iu /= num_fish
            print(ave_needle_acc, ave_needle_iu, ave_fish_acc, ave_fish_iu)
            ave_needle_accs.append(ave_needle_acc)
            ave_needle_ius.append(ave_needle_iu)

            ave_fish_accs.append(ave_fish_acc)
            ave_fish_ius.append(ave_fish_iu)

            PC_Needle_csv_writer.writerow([file_cnt*500, ave_needle_acc])
            JI_Needle_csv_writer.writerow([file_cnt*500, ave_needle_iu])
            PC_Larva_csv_writer.writerow([file_cnt*500, ave_fish_acc])
            JI_Larva_csv_writer.writerow([file_cnt*500, ave_fish_iu])

            file_cnt += 1
        PC_Needle_csv_file.close()
        JI_Needle_csv_file.close()
        PC_Larva_csv_file.close()
        JI_Larva_csv_file.close()
    '''
        line00, = axs[0, 0].plot(epoches, ave_needle_accs, color=color)
        lines_axis00.append(line00)
        line01, = axs[0, 1].plot(epoches, ave_needle_ius, color=color)
        lines_axis01.append(line01)
        line10, = axs[1, 0].plot(epoches, ave_fish_accs, color=color)
        lines_axis10.append(line10)
        line11, = axs[1, 1].plot(epoches, ave_fish_ius, color=color)
        lines_axis11.append(line11)

    axs[0, 0].set_ylabel('PC Needle')
    axs[0, 0].set_xlabel("Training Epoch")
    axs[0, 1].set_ylabel('JI Needle')
    axs[0, 1].set_xlabel("Training Epoch")
    axs[1, 0].set_ylabel('PC Larva')
    axs[1, 0].set_xlabel("Training Epoch")
    axs[1, 1].set_ylabel('JI Larva')
    axs[1, 1].set_xlabel("Training Epoch")

    fig.legend(lines_axis00, model_types, 'upper right')
    fig.legend(lines_axis01, model_types, 'upper right')
    fig.legend(lines_axis10, model_types, 'upper right')
    fig.legend(lines_axis11, model_types, 'upper right')

    #plt.tight_layout()
    plt.show()
    plt.legend(labels=["PC Needle", "JI Needle", "", "JI Larva"], loc="best")
    '''

def test_all_JI_PC(im_anno_list):
    '''
    print("testing binarization")
    test_binarization(im_anno_list)
    print("testing Otsu")
    test_Otsu(im_anno_list)
    print("testing LRB")
    test_LRB(im_anno_list)
    print("testing RG")
    test_RG(im_anno_list)
    '''
    print("testing U-Net")
    test_UNet(im_anno_list)

def test_all_recall_correct_ratio(im_anno_list, thre_steps = 100):
    thresholds = np.arange(thre_steps)/thre_steps
    '''
    print("testing binarization")
    binarization_recall_correct_ratio(im_anno_list, thresholds, save_path="./Methods/Eval-All-Methods/binarization/")
    print("testing Otsu")
    Otsu_recall_correct_ratio(im_anno_list, thresholds, save_path="./Methods/Eval-All-Methods/otsu/")
    print("testing LRB")
    LRB_recall_correct_ratio(im_anno_list, thresholds, save_path="./Methods/Eval-All-Methods/lrb/")
    print("testing RG")
    RG_recall_correct_ratio(im_anno_list, thresholds, save_path="./Methods/Eval-All-Methods/rg/")
    print("testing U-Net")
    '''
    UNet_recall_correct_ratio(im_anno_list, thresholds, save_path="./Methods/Eval-All-Methods/u-net/")

"""
def test_all_recall_false_ratio(im_anno_list, thre_steps = 100):
    threshold = np.arange(thre_steps)/thre_steps
    b_recall_ratios = []
    b_false_ratios = []
    O_recall_ratios = []
    O_false_ratios = []
    L_recall_ratios = []
    L_false_ratios = []
    R_recall_ratios = []
    R_false_ratios = []
    U_recall_ratios = []
    U_false_ratios = []
    for t in threshold:
        print("for threshold:", t)
        r, f = binarization_recall_false_ratio(im_anno_list, t)
        b_recall_ratios.append(r)
        b_false_ratios.append(f)
        print("binarization", r, f)

        r, f = Otsu_recall_false_ratio(im_anno_list, t)
        O_recall_ratios.append(r)
        O_false_ratios.append(f)
        print("Otsu", r, f)

        r, f = LRB_recall_false_ratio(im_anno_list, t)
        L_recall_ratios.append(r)
        L_false_ratios.append(f)
        print("LRB", r, f)

        r, f = RG_recall_false_ratio(im_anno_list, t)
        R_recall_ratios.append(r)
        R_false_ratios.append(f)
        print(r, f)

        r, f = UNet_recall_false_ratio(im_anno_list, t)
        U_recall_ratios.append(r)
        U_false_ratios.append(f)
        print("UNet", r, f)

    fig = plt.figure()
    plt.plot(threshold, b_recall_ratios, marker = ".", label = "Thresholding")
    plt.plot(threshold, O_recall_ratios, marker = "s", label = "Otsu Thresholding")
    plt.plot(threshold, L_recall_ratios, marker = "*", label = "linear regression")
    plt.plot(threshold, R_recall_ratios, marker = "h", label = "Region growing")
    plt.plot(threshold, U_recall_ratios, marker = "x", label = "U Net")
    plt.legend(loc="best")
    plt.xlabel("Threshold of IOU")
    plt.ylabel("Recall Ratio")
    plt.title("Comparison of recall ratio when Threshold of IOU changes")
    plt.show()
    plt.plot(threshold, b_false_ratios, marker = ".", label="Thresholding")
    plt.plot(threshold, O_false_ratios, marker = "s", label="Otsu Thresholding")
    plt.plot(threshold, L_false_ratios, marker = "*", label="linear regression")
    plt.plot(threshold, R_false_ratios, marker = "h", label="Region growing")
    plt.plot(threshold, U_false_ratios, marker = "x", label="U Net")
    plt.legend(loc="best")
    plt.xlabel("Threshold of IOU")
    plt.ylabel("Correct Detection Ratio")
    plt.title("Comparison of correct detection ratio when Threshold of IOU changes")
    plt.show()
"""


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

if __name__ == '__main__':
    test_im_path = "./Methods/UNet_tf/data/test/Images/"
    test_anno_path = "./Methods/UNet_tf/data/test/annotations/"

    im_anno_list = []
    for date in ["01202/", "01203/", "01204/", "01205/"]:
        ims_name = os.listdir(test_im_path + date)
        annos_name = os.listdir(test_anno_path + date)
        for im_name in ims_name:
            name = im_name[:-4]
            im = cv2.imread(test_im_path + date + im_name)
            anno = cv2.imread(test_anno_path + date + name + "_label.tif")
            #anno = cv2.erode(anno, (3, 3), iterations=2)
            anno = anno[:, :, 1]
            anno_needle = np.zeros(anno.shape, dtype=np.uint8)
            anno_needle[np.where(anno == 1)] = 1
            anno_fish = np.zeros(anno.shape, dtype=np.uint8)
            anno_fish[np.where(anno == 2)] = 1

            im_anno_list.append([im, anno_needle, anno_fish])
    print("total images", len(im_anno_list))
    #test_binarization(im_anno_list)
    #test_Otsu(im_anno_list)
    #test_LRB(im_anno_list)
    #(im_anno_list)
    #test_UNet(im_anno_list)
    #test_UNet_detailed(im_anno_list, save=True)
    #test_UNet_select_size_thre(im_anno_list)
    #test_Unet_larva_recall_false_ratio_by_num(im_anno_list, 100)#
    test_all_JI_PC(im_anno_list)
    #test_all_recall_correct_ratio(im_anno_list)
    #test_all_recall_false_ratio(im_anno_list, 20)
    #test_Unet_split_recall_false_ratio(im_anno_list, thre_steps=10)
    #UNet_select_epoch(im_anno_list, modeldir = "Methods/UNet_tf/models_noise/", model_type="Models with augmentation of random Gaussian noise")
    #unet_test.model.load_graph_frozen(model_path="Methods/UNet_tf/ori_UNet/models-trained-on200/models_contrast_finished/UNet500.pb")
    #UNet_select_epoch(im_anno_list)
