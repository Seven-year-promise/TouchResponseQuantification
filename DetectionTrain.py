import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.feature import hog
from skimage.morphology import skeletonize
from xml_reader import XML_Reader
from Methods.LogisticRegression import LogisticRegression
from ImageProcessing import ImageProcessor

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--touching_part', type=str, default = 'tail',
                   help='sum the integers (default: find the max)')

args = parser.parse_args()
touching_index = 0
video_path = ''
datum_paths =['0521-17_00',
              '0605-17_00',
              '0606-17_00',
              '0610-17_00',
              '0611-17_00']
if args.touching_part == 'head':
    video_path = "./detection_test/head/"
    touching_index = 0
elif args.touching_part == 'body':
    video_path = "./detection_test/body/"
    touching_index = 1
elif args.touching_part == 'tail':
    video_path = "./detection_test/tail/"
    touching_index = 2
else:
    print("please select the correct part! ")





if __name__ == '__main__':
    path = './detection_test/Images/'
    xml_base_path = './detection_test/annotation/'
    im_files = os.listdir(path)
    xml_reader = XML_Reader()
    needle_distances = 0
    needle_num = 0
    fish_IOUs = 0
    fish_IOU_list = []
    fish_num = 0
    fish_recall_num = 0
    video_cnt=0
    fish_distances = 0

    images = []
    bboxes = []
    well_infos = []
    LR = LogisticRegression(lr=0.01, num_iter=100000, fit_intercept=True)
    for ipath in im_files:
        video_cnt+=1
        # vpath = 'WT_150931_Speed25.avi'#video_files[10]''
        if ipath[-3:] != 'jpg':
            continue
        #if video_cnt >20:
            #break
        #print(path + ipath)
        xml_file = xml_base_path + ipath[:-3]+'xml'
        xml_reader.file_path = xml_file
        xml_reader.load_file()
        xml_reader.list_objects()
        ground_truth_needles = xml_reader.needles
        #print(ground_truth_needles)
        ground_truth_fishes = xml_reader.fishes

        im = cv2.imread(path + ipath)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_processor = ImageProcessor()
        success, (well_centerx, well_centery, well_radius) = im_processor.well_detection(gray)
        well_infos.append([well_centerx, well_centery, well_radius])

        mask = np.zeros(gray.shape[:2], dtype="uint8")
        cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)

        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

        mask2 = np.ones(gray.shape[:2], dtype="uint8") * 255
        cv2.circle(mask2, (well_centerx, well_centery), well_radius, 0, -1)

        gray_masked += mask2

        #cv2.imshow("gray", gray_masked)
        #cv2.waitKey(30)
        images.append(gray)
        bboxes.append(ground_truth_fishes + xml_reader.needles)
    LR.fit(images, bboxes, well_infos)
    #print(np.round(np.average(np.array(bboxes), axis = 0)))