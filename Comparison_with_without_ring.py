#!/usr/bin/python

"""
Evaluation for image segmentation only by thresholding method.
test on
12 images with ring, 4 larvae
12 images without ring (after shake), 4 larvae
"""

import numpy as np
import time
import os
import csv

import cv2
from Methods.FeatureExtraction import Binarization
from Methods.UNet_tf.test import *
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

binarize = Binarization(method = "Binary")
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
def well_detection(gray, high_thre, low_thre, radius):
    # gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 5,
                               param1=high_thre, param2=low_thre,
                               minRadius=radius-3, maxRadius=radius+3)
    #print(circles)
    '''
    #muted when training
    if circles is not None:
        circles_int = np.uint16(np.around(circles))
        for i in circles_int[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(gray, center, 1, (0, 255, 0), 3)
            # circle outline
            radius = i[2]
            cv2.circle(gray, center, radius, (0, 255, 0), 3)

    cv2.imshow("detected circles", gray)
    cv2.waitKey(1000)
    '''
    if circles is not None:
        well_centerx = np.uint16(np.round(np.average(circles[0, :, 0])))
        well_centery = np.uint16(np.round(np.average(circles[0, :, 1])))
        well_radius = radius #np.uint16(np.round(np.max(circles[0, :, 2])))
        #return True, (well_centerx, well_centery, 110)


    else:
        well_centerx = 240
        well_centery = 240
        well_radius = radius
        #return False, (240, 240, 110)

    # first rough mask for well detection
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    mask_inv = cv2.bitwise_not(mask)
    gray_masked += mask_inv

    return True, (well_centerx, well_centery, well_radius), gray_masked

def binarization(im, thre = 190, type = "no"):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if type == "no":
        _, _, im_well = well_detection(im_gray, 220, 30, 110)
        #cv2.imshow("im_well", im_well)
    else:
        _, _, im_well = well_detection(im_gray, 220, 30, 100)
        #cv2.imshow("im_well", im_well)
    ret, th = cv2.threshold(im_well, thre, 255, cv2.THRESH_BINARY)
    binary = np.zeros(th.shape, np.uint8)
    binary[np.where(th == 0)] = 1
    binary[np.where(th == 255)] = 0

    return binary

def test_binarization(im_anno_list, type):
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno

        binary = binarization(im, type = type)

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

def binarization_recall_correct_ratio(no_ring_im_anno_list, ring_im_anno_list, thresholds):
    num_thre = thresholds.shape[0]
    recall_ratios_ring_no_ring = []
    correct_ratios_ring_no_ring = []
    labels = ["Without ring", "With ring"]

    COLORS = ["tab:green", "tab:red"]
    for t, im_anno_list in enumerate([no_ring_im_anno_list, ring_im_anno_list]):
        num_im = len(im_anno_list)
        recall_ratios = np.zeros((num_im, num_thre), np.float)
        correct_ratios = np.zeros((num_im, num_thre), np.float)

        if t == 0:
            type = "no"
        else:
            type = ""

        for i, im_anno in enumerate(im_anno_list):
            im, anno_needle, anno_fish = im_anno

            binary = binarization(im, type = type)

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
            for j, t in enumerate(thresholds):
                recall_ratio, _, correct_ratio = recall_false_ratio(binary, anno, t,
                                                                              gt_num, gt_blobs)
                recall_ratios[i, j] = recall_ratio
                correct_ratios[i, j] = correct_ratio

        recall_ratios_ave = np.average(recall_ratios, axis=0)
        correct_ratios_ave = np.average(correct_ratios, axis=0)

        recall_ratios_ring_no_ring.append(recall_ratios_ave)
        correct_ratios_ring_no_ring.append(correct_ratios_ave)

    for r, l, c in zip(recall_ratios_ring_no_ring, labels, COLORS):
        plt.plot(thresholds, r, label=l, color=c)

    plt.xlabel("Threshold of IOU ($T_{IOU}$)")
    plt.ylabel("Ratio of recall ($R_r$)")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


    for correct, l, c in zip(correct_ratios_ring_no_ring, labels, COLORS):
        plt.plot(thresholds, correct, label=l, color=c)
    plt.xlabel("Threshold of IOU ($T_{IOU}$)")
    plt.ylabel("Ratio of precision ($R_p$)")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    print("binarization recall and correct ratio, finished")

def load_im(im, type):
    # ---------------- read info -----------------------
    unet_test.ori_im = im
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    unet_test.ori_im_gray = gray

    if type == "no":
        _, (well_x, well_y, _), im_well = well_detection(gray, 220, 30, 110)
        #cv2.imshow("im_well", im_well)
    else:
        _, (well_x, well_y, _), im_well = well_detection(gray, 220, 30, 100)
    #im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

    unet_test.x_min = int(well_x - unet_test.conf.im_size / 2)
    unet_test.x_max = int(well_x + unet_test.conf.im_size / 2)
    unet_test.y_min = int(well_y - unet_test.conf.im_size / 2)
    unet_test.y_max = int(well_y + unet_test.conf.im_size / 2)
    im_block = im_well[unet_test.y_min:unet_test.y_max, unet_test.x_min:unet_test.x_max]

    if DEBUG:
        cv2.imwrite("./Methods/Methods_saved/im_well_block.png", im_block)
    #cv2.imshow("needle", im_block)
    #cv2.waitKey(0)
    img = np.array(im_block, dtype=np.float32)
    img = np.reshape(img, (1, unet_test.conf.im_size, unet_test.conf.im_size, 1))

    img = img / 255 - 0.5
    unet_test.img = np.reshape(img, (1, unet_test.conf.im_size, unet_test.conf.im_size, 1))

def test_UNet(im_anno_list, type):
    unet_test.model.load_graph_frozen(
        model_path="./Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/UNet30000.pb")

    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, _, anno_fish = im_anno

        load_im(im, type)
        _, fish_binary, _, _ = unet_test.predict(threshold=0.9, size=12) # size not used
        binary = np.zeros(fish_binary.shape, np.uint8)
        #binary[np.where(needle_binary > 0)] = 1
        binary[np.where(fish_binary > 0)] = 1

        anno = np.zeros(anno_fish.shape, np.uint8)
        #anno[np.where(anno_needle == 1)] = 1
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

def UNet_recall_correct_ratio(no_ring_im_anno_list, ring_im_anno_list, thresholds):
    unet_test.model.load_graph_frozen(
        model_path="./Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/UNet30000.pb")
    num_thre = thresholds.shape[0]
    recall_ratios_ring_no_ring = []
    correct_ratios_ring_no_ring = []
    labels = ["Without ring", "With ring"]
    LINE_TYPES = ['--', '-.']
    COLORS = ["tab:green", "tab:red"]
    for t, im_anno_list in enumerate([no_ring_im_anno_list, ring_im_anno_list]):
        num_im = len(im_anno_list)
        recall_ratios = np.zeros((num_im, num_thre), np.float)
        correct_ratios = np.zeros((num_im, num_thre), np.float)

        if t == 0:
            type = "no"
        else:
            type = ""

        for i, im_anno in enumerate(im_anno_list):
            im, _, anno_fish = im_anno

            load_im(im, type)
            _, fish_binary, _, _ = unet_test.predict(threshold=0.9, size=12)  # size not used
            binary = np.zeros(fish_binary.shape, np.uint8)
            # binary[np.where(needle_binary > 0)] = 1
            binary[np.where(fish_binary > 0)] = 1

            anno = np.zeros(anno_fish.shape, np.uint8)
            #anno[np.where(anno_needle == 1)] = 1
            anno[np.where(anno_fish == 1)] = 1

            gt_ret, gt_labels = cv2.connectedComponents(anno)
            gt_blobs = get_blobs(gt_ret, gt_labels)
            gt_num = len(gt_blobs)


            #if t>0:
            #   print("type", type, t)
            #    cv2.imshow("binary", binary*255)
            #    cv2.imshow("anno", anno*255)
            #    cv2.waitKey(0)
            recall_ratio_list = []
            recall_ratio_list.append(i)
            correct_ratio_list = []
            correct_ratio_list.append(i)
            for j, th in enumerate(thresholds):
                recall_ratio, _, correct_ratio = recall_false_ratio(binary, anno, th,
                                                                              gt_num, gt_blobs)
                recall_ratios[i, j] = recall_ratio
                correct_ratios[i, j] = correct_ratio

        recall_ratios_ave = np.average(recall_ratios, axis=0)
        correct_ratios_ave = np.average(correct_ratios, axis=0)

        recall_ratios_ring_no_ring.append(recall_ratios_ave)
        correct_ratios_ring_no_ring.append(correct_ratios_ave)


    axis_font = {'fontname': 'Times New Roman', 'size': '18'}
    legend_font = font_manager.FontProperties(family='Times New Roman',
                                       style='normal', size=18)

    for r, l, c, l_t in zip(recall_ratios_ring_no_ring, labels, COLORS, LINE_TYPES):
        plt.plot(thresholds, r, ls = l_t, label=l, color=c)

    plt.xlabel("Threshold of IOU ($T_{IOU}$)", **axis_font)
    plt.ylabel("Ratio of recall ($R_r$)", **axis_font)
    plt.ylim((0, 1))
    plt.legend(loc="upper right", prop=legend_font)
    plt.tight_layout()
    plt.show()


    for correct, l, c, l_t in zip(correct_ratios_ring_no_ring, labels, COLORS, LINE_TYPES):
        plt.plot(thresholds, correct, ls = l_t, label=l, color=c)
    plt.xlabel("Threshold of IOU ($T_{IOU}$)", **axis_font)
    plt.ylabel("Ratio of precision ($R_p$)", **axis_font)
    plt.ylim((0, 1))
    plt.legend(loc="upper right", prop=legend_font)
    plt.tight_layout()
    plt.show()
    print("binarization recall and correct ratio, finished")


def test_all_JI_PC(no_ring_im_anno_list, ring_im_anno_list):
    print("data for no ring")
    #test_binarization(no_ring_im_anno_list, type = "no")
    test_UNet(no_ring_im_anno_list, type="no")

    print("data for ring")
    #test_binarization(ring_im_anno_list, type = "")
    test_UNet(ring_im_anno_list, type="")

def test_all_recall_correct_ratio(no_ring_im_anno_list, ring_im_anno_list, thre_steps = 100):
    thresholds = np.arange(thre_steps - 1)/thre_steps + 0.01

    print("testing U-Net")
    #+-binarization_recall_correct_ratio(no_ring_im_anno_list, ring_im_anno_list, thresholds)

    UNet_recall_correct_ratio(no_ring_im_anno_list, ring_im_anno_list, thresholds)

if __name__ == '__main__':
    no_ring_im_path = "./comparison_with_without_ring/20210611-white-no-rings-no-touching/Images/"
    no_ring_anno_path = "./comparison_with_without_ring/20210611-white-no-rings-no-touching/Annotations/"

    ring_im_path = "./comparison_with_without_ring/20210521-5s-spontaneous-movement/after_shake/Images/"
    ring_anno_path = "./comparison_with_without_ring/20210521-5s-spontaneous-movement/after_shake/Annotations/"


    no_ring_im_anno_list = []
    ims_name = os.listdir(no_ring_im_path)
    annos_name = os.listdir(no_ring_anno_path)
    for im_name in ims_name:
        name = im_name[:-4]
        im = cv2.imread(no_ring_im_path + im_name)
        anno = cv2.imread(no_ring_anno_path + name + "_label.tif")
        #anno = cv2.erode(anno, (3, 3), iterations=2)
        anno = anno[:, :, 1]
        anno_needle = np.zeros(anno.shape, dtype=np.uint8)
        anno_needle[np.where(anno == 1)] = 1
        anno_fish = np.zeros(anno.shape, dtype=np.uint8)
        anno_fish[np.where(anno == 2)] = 1

        no_ring_im_anno_list.append([im, anno_needle, anno_fish])
    print("total images", len(no_ring_im_anno_list))

    ring_im_anno_list = []
    ims_name = os.listdir(ring_im_path)
    annos_name = os.listdir(ring_anno_path)
    for im_name in ims_name:
        name = im_name[:-4]
        im = cv2.imread(ring_im_path + im_name)
        anno = cv2.imread(ring_anno_path + name + "_label.tif")
        # anno = cv2.erode(anno, (3, 3), iterations=2)
        anno = anno[:, :, 1]
        anno_needle = np.zeros(anno.shape, dtype=np.uint8)
        anno_needle[np.where(anno == 1)] = 1
        anno_fish = np.zeros(anno.shape, dtype=np.uint8)
        anno_fish[np.where(anno == 2)] = 1

        ring_im_anno_list.append([im, anno_needle, anno_fish])
    print("total images", len(ring_im_anno_list))

    #test_all_JI_PC(no_ring_im_anno_list, ring_im_anno_list)
    test_all_recall_correct_ratio(no_ring_im_anno_list, ring_im_anno_list, 100)

