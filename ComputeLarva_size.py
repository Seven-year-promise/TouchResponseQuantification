#!/usr/bin/python

"""
compute the average number of pixels for the larvae on Day 3
the average is 162.66 pixels
"""

from Methods.UNet_tf.test import *
import cv2

def compute_size(anno_im):
    eval_ret, eval_labels = cv2.connectedComponents(anno_im)
    sizes = np.zeros(eval_ret-1)
    for n in range(1, eval_ret):
        coordinate = np.where(eval_labels == n)
        print(coordinate[0].shape)
        sizes[n-1] = coordinate[0].shape[0]

    ave_size = np.average(sizes)
    max_size = np.max(sizes)

    return ave_size, max_size

if __name__ == '__main__':
    test_im_path = "./Methods/UNet_tf/data/test/Images/"
    test_anno_path = "./Methods/UNet_tf/data/test/annotations/"

    im_anno_list = []
    max_sizes = []
    ave_sizes = []
    for date in ["01202/", "01203/", "01204/", "01205/"]:
        ims_name = os.listdir(test_im_path + date)
        annos_name = os.listdir(test_anno_path + date)
        for im_name in ims_name:
            name = im_name[:-4]
            im = cv2.imread(test_im_path + date + im_name)
            anno = cv2.imread(test_anno_path + date + name + "_label.tif")
            #anno = cv2.erode(anno, (3, 3), iterations=2)
            anno = anno[:, :, 1]
            anno_fish = np.zeros(anno.shape, dtype=np.uint8)
            anno_fish[np.where(anno == 2)] = 1

            ave_s, max_s = compute_size(anno_fish)

            ave_sizes.append(ave_s)
            max_sizes.append(max_s)

    ave_larva_size = np.average(ave_sizes)
    max_larva_size = np.max(max_sizes)

    print("larva average size:", ave_larva_size, "larva maximum size:", max_larva_size)
