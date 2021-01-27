#!/usr/bin/python

"""
Evaluation for image segmentation.
"""

import numpy as np
import time
import os
from Methods.UNet_tf.test import *
import cv2
from scipy.spatial import distance
from Methods.FeatureExtraction import Binarization
from Methods.ImageProcessing import well_detection

import matplotlib.pyplot as plt

unet_test = UNetTestTF()

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

def recall_false_ratio(eval_segm, gt_segm, threshold, larva_num = 5):
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
        false_ratio = CP / (eval_num)
        return recall_ratio, false_ratio
    else:
        return None, None

def dice_loss(pred, target, smooth=1.):
    intersection = np.logical_and(pred, target)
    return 1- (2 * intersection.sum() + smooth)/ (pred.sum() + target.sum() + smooth)


def UNet_select_epoch(im_anno_list, modeldir, model_type = "Models without augmentation"):
    model_files = [f for f in os.listdir(modeldir) if f.endswith('.pb')]

    def model_num(x):
        return (int(x[4:-3]))

    sorted_files = sorted(model_files, key=model_num)

    file_num = len(sorted_files)
    epoches = np.arange(1, file_num+1)*500

    ave_losses = []
    print(sorted_files)
    for m_f in sorted_files:
        print(m_f)
        unet_test.model.load_graph(model_path=modeldir + m_f)
        ave_loss = 0
        num_im = len(im_anno_list)
        i = 0
        for im_anno in im_anno_list:
            i += 1
            im, anno_needle, anno_fish = im_anno

            unet_test.load_im(im)
            needle_binary, fish_binary, fish_points = unet_test.predict(threshold=0.9, size=44)

            prediction = np.array((needle_binary, fish_binary))
            target = np.array((anno_needle, anno_fish))

            ave_loss += dice_loss(prediction, target)
            # cv2.imshow("binary", binary*255)
            # cv2.waitKey(0)
            # cv2.imshow("anno", anno*255)
            # cv2.waitKey(0)
        ave_loss /= num_im
        print(m_f, ave_loss)
        ave_losses.append(ave_loss)

    plt.plot(epoches, ave_losses, marker="*")

    plt.legend(labels=["Dice loss for evaluation dataset"], loc="best")
    plt.xlabel("Training Epoch")
    plt.ylabel("Dice Loss")
    plt.title(model_type)
    plt.show()

'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

if __name__ == '__main__':
    base_im_path = "Methods/UNet_tf/data/test/Images/"
    base_anno_path = "Methods/UNet_tf/data/test/annotations/"
    im_anno_list = []
    for fish_num in range(2, 6):
        test_im_path = base_im_path + str(fish_num) + "/"
        test_anno_path = base_anno_path + str(fish_num) + "/"
        ims_name = os.listdir(test_im_path)
        annos_name = os.listdir(test_anno_path)

        for im_name in ims_name:
            name = im_name[:-4]
            im = cv2.imread(test_im_path + im_name)
            anno = cv2.imread(test_anno_path + name + "_label.tif")
            #anno = cv2.erode(anno, (3, 3), iterations=2)
            anno = anno[:, :, 1]
            anno_needle = np.zeros(anno.shape, dtype=np.uint8)
            anno_needle[np.where(anno == 1)] = 1
            anno_fish = np.zeros(anno.shape, dtype=np.uint8)
            anno_fish[np.where(anno == 2)] = 1

            im_anno_list.append([im, anno_needle, anno_fish])

    UNet_select_epoch(im_anno_list[::100], modeldir = "Methods/UNet_tf/models_rotate_contrast_finished/", model_type="Models with augmentation of random rotation, contrast and brightness")