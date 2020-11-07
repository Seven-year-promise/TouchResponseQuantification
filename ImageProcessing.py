import cv2
import numpy as np
from Methods.FeatureExtraction import SIFT, Binarization


class ImageProcessor:
    def __init__(self):
        self.sift = SIFT()
        self.binarize = Binarization()

    def well_detection(self, gray):
        # gray = cv2.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 20,
                                   param1=240, param2=50,
                                   minRadius=80, maxRadius=90)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(gray, center, 1, (0, 255, 0), 3)
                # circle outline
                radius = i[2]
                cv2.circle(gray, center, radius, (0, 255, 0), 3)
        # cv2.imshow("detected circles", gray)
        # cv2.waitKey(0)
        if circles is not None:
            well_centerx = np.uint16(np.round(np.average(circles[0, :, 0])))
            well_centery = np.uint16(np.round(np.average(circles[0, :, 1])))
            well_radius = np.uint16(np.round(np.average(circles[0, :, 2]) * 0.9))
            return True, (well_centerx, well_centery, well_radius)
        else:
            return False, (240, 240, 70)


    def feature_extraction(self, ori_im, method = "Otsu", well_infos = None):
        #self.sift.detect_keypoints(ori_im)
        #im_with_keypoints = self.sift.drawpoints(ori_im)
        if method == "sift":
            im_feature = self.sift.compute_sift(ori_im)
        elif method == "Otsu":
            self.binarize.method = method
            im_feature = self.binarize.compute_binary(ori_im)
        elif method == "LRB":
            self.binarize.method = method
            im_feature = self.binarize.compute_binary(ori_im, well_infos)
        return im_feature, None, None #im_with_keypoints, self.sift.keypoints, self.sift.descriptors

    def meanshift_seg(self, ori_im):
        # TO DO
        pass

