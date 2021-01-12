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

binarize = Binarization(method = "Binary")
ostu = Binarization(method = "Otsu")
lrb = Binarization(method = "LRB")
rg = Binarization(method = "RG")
unet_test = UNetTestTF()
unet_test.model.load_graph(model_path="Methods/UNet_tf/models/UNet18000.pb")

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

def get_roi(blobA, blobB, ori_shape):
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

def recall_false_ratio(eval_segm, gt_segm, threshold):
    '''
    recall_ratio: TP / (TP + TN)
    false_ratio: FP / (TP + FP)
    correct_ratio: CP / (TP + FP)
    '''

    check_size(eval_segm, gt_segm)

    eval_ret, eval_labels = cv2.connectedComponents(eval_segm)
    #cv2.imshow("label", np.array(eval_labels*(255/eval_ret), np.uint8))
    gt_ret, gt_labels = cv2.connectedComponents(gt_segm)
    eval_blobs = get_blobs(eval_ret, eval_labels)
    gt_blobs = get_blobs(gt_ret, gt_labels)

    eval_num = len(eval_blobs)
    gt_num = len(gt_blobs)
    #print("BEGIN", gt_ret, eval_ret)
    eval_found_flag = np.zeros(eval_num, np.uint8)
    gt_found_flag = np.zeros(gt_num, np.uint8)

    for g_n in range(gt_num):
        gt_blob = gt_blobs[g_n]
        for e_n in range(eval_num):
            eval_blob = eval_blobs[e_n]
            roi = get_roi(gt_blob, eval_blob, eval_segm.shape)
            #print("roi", roi)
            if roi > threshold:
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

def binarization_recall_false_ratio(im_anno_list, threshold):

    ave_recall_ratio = 0
    ave_false_ratio = 0
    num = len(im_anno_list)

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

        recall_ratio, false_ratio = recall_false_ratio(binary, anno, threshold)
        ave_recall_ratio += recall_ratio
        ave_false_ratio += false_ratio


    return ave_recall_ratio/num, ave_false_ratio/num

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
        binary = ostu.Otsu(im_well)

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

def Otsu_recall_false_ratio(im_anno_list, threshold):
    ave_recall_ratio = 0
    ave_false_ratio = 0
    num = len(im_anno_list)

    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, im_gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

        binary = ostu.Otsu(im_well)

        anno = np.zeros(anno_needle.shape, np.uint8)
        anno[np.where(anno_needle == 1)] = 1
        anno[np.where(anno_fish == 1)] = 1
        #cv2.imshow("binary", binary*255)
        #cv2.waitKey(0)
        #cv2.imshow("anno", anno*255)
        #cv2.waitKey(0)
        recall_ratio, false_ratio = recall_false_ratio(binary, anno, threshold)
        ave_recall_ratio += recall_ratio
        ave_false_ratio += false_ratio

    return ave_recall_ratio / num, ave_false_ratio / num

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

def LRB_recall_false_ratio(im_anno_list, threshold):

    ave_recall_ratio = 0
    ave_false_ratio = 0
    num = len(im_anno_list)

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

        recall_ratio, false_ratio = recall_false_ratio(binary, anno, threshold)
        ave_recall_ratio += recall_ratio
        ave_false_ratio += false_ratio

    return ave_recall_ratio / num, ave_false_ratio / num

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

def RG_recall_false_ratio(im_anno_list, threshold):
    ave_recall_ratio = 0
    ave_false_ratio = 0
    num = len(im_anno_list)

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
        recall_ratio, false_ratio = recall_false_ratio(binary, anno, threshold)
        ave_recall_ratio += recall_ratio
        ave_false_ratio += false_ratio

    return ave_recall_ratio / num, ave_false_ratio / num

def test_UNet(im_anno_list):

    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno

        unet_test.load_im(im)
        needle_binary, fish_binary = unet_test.predict(threshold=0.9)
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

def UNet_recall_false_ratio(im_anno_list, threshold):
    ave_recall_ratio = 0
    ave_false_ratio = 0
    num = len(im_anno_list)

    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno

        unet_test.load_im(im)
        needle_binary, fish_binary, _ = unet_test.predict(threshold=0.9)
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
        recall_ratio, false_ratio = recall_false_ratio(binary, anno, threshold)
        ave_recall_ratio += recall_ratio
        ave_false_ratio += false_ratio

    return ave_recall_ratio / num, ave_false_ratio / num

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
        needle_binary, fish_binary, im_with_points, fish_points = unet_test.get_keypoint(threshold=0.9, size_fish=44)

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

def test_UNet_select_size_thre(im_anno_list, save = False):

    ave_needle_accs = []
    ave_fish_accs = []
    ave_needle_ius = []
    ave_fish_ius = []
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
            needle_binary, fish_binary, im_with_points, fish_points = unet_test.get_keypoint(threshold=0.9, size_fish=threshold)

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
        print("average needle accuracy", ave_needle_acc)
        print("average needle iu", ave_needle_iu)

        print("average fish accuracy", ave_fish_acc)
        print("average fish iu", ave_fish_iu)
        ave_needle_accs.append(ave_needle_acc)
        ave_needle_ius.append(ave_needle_iu)
        ave_fish_accs.append(ave_fish_acc)
        ave_fish_ius.append(ave_fish_iu)
    plt.plot(ave_fish_accs)
    plt.plot(ave_fish_ius)
    plt.show()
    time_used = time.time() - time_cnt


    print("time per frame", time_used / num_im)

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
    plt.plot(b_recall_ratios, label = "Thresholding")
    plt.plot(O_recall_ratios, label = "Ostu Thresholding")
    plt.plot(L_recall_ratios, label = "linear regression")
    plt.plot(R_recall_ratios, label = "Region growing")
    plt.plot(U_recall_ratios, label = "U Net")
    plt.legend(loc="best")
    plt.show()
    plt.plot(b_false_ratios, label="Thresholding")
    plt.plot(O_false_ratios, label="Ostu Thresholding")
    plt.plot(L_false_ratios, label="linear regression")
    plt.plot(R_false_ratios, label="Region growing")
    plt.plot(U_false_ratios, label="U Net")
    plt.legend(loc="best")
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
    test_im_path = "Methods/LightUNet/dataset/test/Images/"
    test_anno_path = "Methods/LightUNet/dataset/test/annotation/"
    ims_name = os.listdir(test_im_path)
    annos_name = os.listdir(test_anno_path)
    im_anno_list = []
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

    #test_binarization(im_anno_list)
    #test_Otsu(im_anno_list)
    #test_LRB(im_anno_list)
    #(im_anno_list)
    #test_UNet(im_anno_list)
    #test_UNet_detailed(im_anno_list, save=True)
    #test_UNet_select_size_thre(im_anno_list)
    test_all_recall_false_ratio(im_anno_list[:3], 2)