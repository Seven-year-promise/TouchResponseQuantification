#!/usr/bin/python

"""
Evaluation for image segmentation.
"""

import numpy as np
import time
import os
from Methods.UNet_tf.test import UNetTestTF
import cv2
from Methods.FeatureExtraction import Binarization
from Methods.ImageProcessing import well_detection

import matplotlib.pyplot as plt


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


def load_im(im_anno_patch, im_size = 240):
    # ---------------- read info -----------------------
    im_num = len(im_anno_patch)
    out_im_patch = np.zeros((im_num, im_size, im_size, 1), np.float32)
    out_anno_patch = np.zeros((im_num, im_size, im_size, 2), np.float32)
    for n in range(im_num):
        im, anno_needle, anno_fish = im_anno_patch[n]

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        _, (well_x, well_y, _), im_well = well_detection(im, gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

        x_min = int(well_x - im_size / 2)
        x_max = int(well_x + im_size / 2)
        y_min = int(well_y - im_size / 2)
        y_max = int(well_y + im_size / 2)
        im_block = im_well[y_min:y_max, x_min:x_max]
        #cv2.imshow("needle", im_block)
        #cv2.waitKey(0)
        img = np.array(im_block, dtype=np.float32)
        img = np.reshape(img, (1, im_size, im_size, 1))

        img = img / 255 - 0.5
        img = np.reshape(img, (1, im_size, im_size, 1))
        out_im_patch[n, :, :, :] = img

        anno_needle_block = anno_needle[y_min:y_max, x_min:x_max]
        anno_fish_block = anno_fish[y_min:y_max, x_min:x_max]
        out_anno_patch[n, :, :, 0] = anno_needle_block
        out_anno_patch[n, :, :, 1] = anno_fish_block

    return out_im_patch, out_anno_patch

def eval(seg_patch, anno_patch):
    ave_needle_acc = 0
    ave_fish_acc = 0
    ave_needle_iu = 0
    ave_fish_iu = 0
    num_needle = 0
    num_fish = 0
    num_im = len(seg_patch)
    time_cnt = time.time()
    i = 0
    for n in range(num_im):
        i += 1
        needle_binary = seg_patch[n, :, :, 0]
        fish_binary = seg_patch[n, :, :, 1]
        anno_needle = anno_patch[n, :, :, 0]
        anno_fish = anno_patch[n, :, :, 1]

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

    return ave_needle_acc / num_needle, \
           ave_fish_acc / num_fish, \
           ave_needle_iu / num_needle, \
           ave_fish_iu / num_fish


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
