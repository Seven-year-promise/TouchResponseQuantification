import cv2
import numpy as np
import os
import argparse
import time
import math

from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert

from Methods.Curvature import ComputeCurvature


class QuantificationIndex:
    """
    given needle points, larva points and larva patches,
    compute the quantification indexes, latency time (t_l), C-Shape Radius Average (r_ca), Moving distance (d_m), response time (t_r)
    importance time points: t1: touch apllied, t2: response begins, t3: response stops
    """
    def __init__(self, n_l_dis_thre, move_thre):
        self.needle_larva_dis_thre = n_l_dis_thre
        self.move_thre = move_thre
        self.needle_points = None
        self.larva_pointss = None
        self.larva_patchess = None
        self.larva_touched = 0
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.frame_length = 0
        self.ComputeCur = ComputeCurvature(degree=3)

    def compute_t1(self, larva_first_centers):
        """
        compute the frame time for touching applied
        :return: the time point of t1, or -1 if not touched successfully
        """
        found_flag = False
        p_t0 = larva_first_centers[self.larva_touched]
        for t, (n_p, l_p) in enumerate(zip(self.needle_points, self.larva_pointss)):
            d = math.sqrt((n_p[0] - p_t0[0]) ** 2 + (n_p[1] - p_t0[1]) ** 2)
            if d < self.needle_larva_dis_thre:
                found_flag = True
                return t

        if not found_flag:
            return -1

    def compute_t2(self):
        """
        get the time of response begins: t2
        :return: t2, or -1: not moved
        """
        found_flag = False
        p_t0 = self.larva_pointss[self.t1][self.larva_touched]
        for t in range(self.t1, self.frame_length):
            p_t1 = self.larva_pointss[t][self.larva_touched]
            d = math.sqrt((p_t0[0] - p_t1[0]) ** 2 + (p_t0[1] - p_t1[1]) ** 2)
            if d > self.move_thre:
                found_flag = True
                return t

        if not found_flag:
            return -1

    def compute_t3(self):
        """
        get the time of response stops: t3
        :return: t3, or -1: errors, as there must one frame with response stops if self.t2 is not -1
        """
        found_flag = False
        p_last = self.larva_pointss[-1][self.larva_touched]
        # check from the last one
        for i in range(1, self.frame_length - self.t2):
            t = self.frame_length - i
            p_t1 = self.larva_pointss[t-1][self.larva_touched]
            d = math.sqrt((p_last[0] - p_t1[0]) ** 2 + (p_last[1] - p_t1[1]) ** 2)
            if d > self.move_thre:
                found_flag = True
                return t - 1

        if not found_flag:
            return -1

    def compute_radius(self, larva_patch):
        """
        compute the C-Shape radius for the larva patch
        :param larva_patch:
        :return:
        """
        skeleton = skeletonize(larva_patch)

        skeleton_cor = np.where(skeleton > 0)
        (skeleton_y, skeleton_x) = (skeleton_cor[0], skeleton_cor[1])
        if len(skeleton_x) < 4:
            return -1
        else:
            skeleton_minx = np.min(skeleton_x)
            skeleton_miny = np.min(skeleton_y)
            skeleton_maxx = np.max(skeleton_x)
            skeleton_maxy = np.max(skeleton_y)

            radius = self.ComputeCur.non_linear_fit(skeleton_x, skeleton_y)
            return radius

    def compute_r_ca_d_m(self):
        distance = 0
        radiuses = []
        # print(all_points)
        for i in range(self.t2, self.t3 + 1):
            p1 = self.larva_pointss[i][self.larva_touched]
            p2 = self.larva_pointss[i + 1][self.larva_touched]
            d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            distance += d
            l_patch = self.larva_patchess[i][self.larva_touched]
            radius = self.compute_radius(l_patch)
            if (radius >= 0) and (radius != "nan"):
                radiuses.append(radius)
            # print(p1, p2, d, distance)
        # print(distance)
        r_ave = np.average(np.array(radiuses))
        return distance, r_ave

    def get_indexes(self, larva_first_centers, needle_points, larva_pointss, larva_patchess, larva_touched):
        """

        :param needle_points:
        :param larva_pointss: the larva points for all the frames, must be in "float" without np.round
        :param larva_patches:
        :param larva_touched:
        :return:
        """
        self.needle_points = needle_points
        self.larva_pointss = larva_pointss
        self.larva_patchess = larva_patchess
        self.larva_touched = larva_touched
        self.frame_length = len(self.needle_points)
        self.t1 = self.compute_t1(larva_first_centers)
        self.t2 = self.compute_t2()
        self.t3 = self.compute_t3()

        t_l = (self.t2 - self.t1 + 1) / 1000.0
        t_r = (self.t3 - self.t2 + 1) / 1000.0

        d_m, r_ca = self.compute_r_ca_d_m()

        return t_l, r_ca, t_r, d_m