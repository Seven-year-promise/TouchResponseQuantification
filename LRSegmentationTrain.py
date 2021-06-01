import argparse
import cv2
import numpy as np
import os
from xml_reader import XML_Reader
from Methods.LogisticRegression import LogisticRegression, LogisticRegressionSeg
from Methods.UNet_tf.util import well_detection

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
    video_path = "./detection_train/head/"
    touching_index = 0
elif args.touching_part == 'body':
    video_path = "./detection_train/body/"
    touching_index = 1
elif args.touching_part == 'tail':
    video_path = "./detection_train/tail/"
    touching_index = 2
else:
    print("please select the correct part! ")





if __name__ == '__main__':
    train_im_path = "./Methods/UNet_tf/data/train/Images/"
    train_anno_path = "./Methods/UNet_tf/data/train/annotation/"

    LR = LogisticRegressionSeg(lr=[0.01, 0.0001], model_save_path="./LR_models/train-on200",
                               num_iter=1000000, fit_intercept=True)

    images = []
    annos = []
    for date in ["2020/"]:
        ims_name = os.listdir(train_im_path + date)
        annos_name = os.listdir(train_anno_path + date)
        print("loading images .....")
        for im_name in ims_name:
            name = im_name[:-4]
            print("loading image ", name, end='\r')
            # for preprocessing of images
            im = cv2.imread(train_im_path + date + im_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            success, (well_centerx, well_centery, well_radius), im_well = well_detection(im, gray)
            im_well_gray = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
            im_well_gray = im_well_gray[(well_centery-well_radius):(well_centery+well_radius), (well_centerx-well_radius):(well_centerx+well_radius)]
            images.append(im_well_gray)

            # open annotation
            anno = cv2.imread(train_anno_path + date + name + "_label.tif")
            # anno = cv2.erode(anno, (3, 3), iterations=2)
            anno = anno[:, :, 1]
            anno_im = np.zeros(anno.shape, dtype=np.uint8)
            anno_im[np.where(anno == 1)] = 1
            anno_im[np.where(anno == 2)] = 1
            anno_im = anno_im[(well_centery-well_radius):(well_centery+well_radius), (well_centerx-well_radius):(well_centerx+well_radius)]
            annos.append(anno_im)
    print("loading", len(images), "images", "finished")
    LR.fit(images, annos, batch_size=128)

    """
    path = './detection_train/multiFish/training/Images/'
    xml_base_path = './detection_train/multiFish/training/annotation/'
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
    LR = LogisticRegression(lr=[0.01, 0.0001], num_iter=1000000, fit_intercept=True)
    #LR = LogisticRegressionTorch(lr=[0.01, 0.0001], resume=False, num_iter=1000000, fit_intercept=True)
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
        images.append(gray_masked)
        bboxes.append(ground_truth_fishes + xml_reader.needles)
    #print(np.round(np.average(np.array(bboxes), axis=0)))
    LR.fit(images, bboxes, well_infos)
    #print(np.round(np.average(np.array(bboxes), axis = 0)))
    """