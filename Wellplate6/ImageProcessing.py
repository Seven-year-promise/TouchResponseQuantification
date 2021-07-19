import numpy as np
import time
import os
import csv

import cv2


import matplotlib.pyplot as plt

def well_detection(gray, high_thre, low_thre, radius):
    # gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 5,
                               param1=high_thre, param2=low_thre,
                               minRadius=radius-10, maxRadius=radius+10)
    #print(circles)
    radius = 175

    #muted when training
    """
    im_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles_int = np.uint16(np.around(circles))
        for i in circles_int[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(im_color, center, 1, (0, 255, 0), 3)
            # circle outline
            radius = i[2]
            cv2.circle(im_color, center, 170, (0, 255, 0), 3)
    cv2.imshow("detected circles", im_color)
    cv2.waitKey(0)
    """

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

    cv2.imshow("cropped", gray_masked)
    cv2.waitKey(0)

    return True, (well_centerx, well_centery, well_radius), gray_masked

def binarization(im, thre = 190, type = "no"):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if type == "no":
        _, _, im_well = well_detection(im_gray, 220, 30, 200)
        #cv2.imshow("im_well", im_well)
    else:
        _, _, im_well = well_detection(im_gray, 220, 30, 200)
        #cv2.imshow("im_well", im_well)
    ret, th = cv2.threshold(im_well, thre, 255, cv2.THRESH_BINARY)
    binary = np.zeros(th.shape, np.uint8)
    binary[np.where(th == 0)] = 1
    binary[np.where(th == 255)] = 0

    return binary

if __name__ == "__main__":
    path = "./20210709-6-well-images_no_larvae/"
    im_paths = os.listdir(path)
    for im_name in im_paths:
        print(path + im_name)
        im = cv2.imread(path + im_name)
        binarization(im)

