#!/usr/bin/python

"""
Evaluation for image segmentation.
"""

import numpy as np
import time
import os
from Methods.LightUNet.UNet import UNet
from Methods.LightUNet.test import UNetTest
import cv2
from Methods.FeatureExtraction import Binarization
from Methods.ImageProcessing import well_detection

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
    print(cl)

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

def test_binarization(im_anno_list):
    binarize = Binarization(method = "Binary")
    one_frame = im_anno_list[0][0]
    one_frame_gray = cv2.cvtColor(one_frame, cv2.COLOR_BGR2GRAY)
    success, (well_centerx, well_centery, well_radius) = well_detection(one_frame_gray)

    mask = np.zeros(one_frame_gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)
    mask2 = np.ones(one_frame_gray.shape[:2], dtype="uint8") * 255
    cv2.circle(mask2, (well_centerx, well_centery), well_radius, 0, -1)
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(im_gray, im_gray, mask=mask)
        gray_masked += mask2
        binary = binarize.Binary(gray_masked, needle_thr=180)

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

def test_Otsu(im_anno_list):
    binarize = Binarization(method = "Otsu")
    one_frame = im_anno_list[0][0]
    one_frame_gray = cv2.cvtColor(one_frame, cv2.COLOR_BGR2GRAY)
    success, (well_centerx, well_centery, well_radius) = well_detection(one_frame_gray)

    mask = np.zeros(one_frame_gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)
    mask2 = np.ones(one_frame_gray.shape[:2], dtype="uint8") * 255
    cv2.circle(mask2, (well_centerx, well_centery), well_radius, 0, -1)
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(im_gray, im_gray, mask=mask)
        gray_masked += mask2
        binary = binarize.Otsu(gray_masked)

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

def test_LRB(im_anno_list):
    binarize = Binarization(method = "LRB")
    one_frame = im_anno_list[0][0]
    one_frame_gray = cv2.cvtColor(one_frame, cv2.COLOR_BGR2GRAY)
    success, (well_centerx, well_centery, well_radius) = well_detection(one_frame_gray)

    mask = np.zeros(one_frame_gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)
    mask2 = np.ones(one_frame_gray.shape[:2], dtype="uint8") * 255
    cv2.circle(mask2, (well_centerx, well_centery), well_radius, 0, -1)
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(im_gray, im_gray, mask=mask)
        gray_masked += mask2
        binary = binarize.LRB(gray_masked, well_infos=(well_centerx, well_centery, well_radius))

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

def test_RG(im_anno_list):
    binarize = Binarization(method = "RG")
    one_frame = im_anno_list[0][0]
    one_frame_gray = cv2.cvtColor(one_frame, cv2.COLOR_BGR2GRAY)
    success, (well_centerx, well_centery, well_radius) = well_detection(one_frame_gray)

    mask = np.zeros(one_frame_gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)
    mask2 = np.ones(one_frame_gray.shape[:2], dtype="uint8") * 255
    cv2.circle(mask2, (well_centerx, well_centery), well_radius, 0, -1)
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(im_gray, im_gray, mask=mask)
        gray_masked += mask2
        binary = binarize.RG(gray_masked, threshold = 5)

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

def test_UNet(im_anno_list):
    unet_test = UNetTest(n_class=2, cropped_size=240, model_path="Methods/LightUNet/6000.pth.tar")
    unet_test.load_model()
    ave_acc = 0
    ave_iu = 0
    num = len(im_anno_list)
    time_cnt = time.time()
    for im_anno in im_anno_list:
        im, anno_needle, anno_fish = im_anno

        unet_test.load_im(im)
        binary = unet_test.predict(threshold=0.9)
        binary[np.where(binary > 0)] = 1

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

def test_UNet_detailed(im_anno_list, save = True):
    unet_test = UNetTest(n_class=2, cropped_size=240, model_path="Methods/LightUNet/6000.pth.tar")
    unet_test.load_model()
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
        needle_binary, fish_binary = unet_test.get_keypoint(threshold=0.9)

        if save:
            save_im = np.array(needle_binary.shape, np.uint8)
            save_im[np.where(needle_binary == 1)] = 1
            save_im[np.where(fish_binary == 1)] = 2
            cv2.imwrite("GUI_saved/" + str(i) + "ori.jpg", im)
            cv2.imwrite("GUI_saved/" + str(i) + "binary.jpg", save_im*127)

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
        anno = cv2.erode(anno, (3, 3), iterations=2)
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
    test_UNet_detailed(im_anno_list, save=False)
