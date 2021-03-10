from Methods.UNet_tf.util import *
import numpy as np
from skimage.morphology import skeletonize
import math

import tensorflow as tf
from Methods.UNet_tf.UNet import *
DEBUG = True

def configure():
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 20000, 'How many steps to train')
    flags.DEFINE_float('rate', 1e-4, 'learning rate for training')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('save_interval', 2000, 'interval to save model')
    flags.DEFINE_integer('summary_interval', 100, 'interval to save summary')
    flags.DEFINE_integer('n_classes', 2, 'output class number')
    flags.DEFINE_integer('batch_size', 12, 'batch size for one iter')
    flags.DEFINE_boolean('is_training', True, 'training or predict (for batch normalization)')
    flags.DEFINE_string('datadir', './data/train/train.tfrecords', 'path to tfrecords')
    flags.DEFINE_string('logdir', 'logs', 'directory to save logs of accuracy and loss')
    flags.DEFINE_string('modeldir', 'models', 'directory to save models ')
    flags.DEFINE_string('model_name', 'UNet', 'Model file name')
    flags.DEFINE_integer('ori_size', 480, 'the sie of input image')
    flags.DEFINE_integer('im_size', 240, 'the sie of training image')
    flags.DEFINE_float('bce_weight', 0.5, 'weight for loss')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

class UNetTestTF:
    def __init__(self):
        self.conf = configure()
        self.model = UNet(tf.Session(), self.conf)
        self.img = None

    def load_im(self, im):
        # ---------------- read info -----------------------
        self.ori_im = im
        if DEBUG:
            cv2.imwrite("./Methods/Methods_saved/ori_im.png", im)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.ori_im_gray = gray
        _, (well_x, well_y, _), im_well = well_detection(im, gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        if DEBUG:
            cv2.imwrite("./Methods/Methods_saved/im_well.png", im_well)
        self.x_min = int(well_x - self.conf.im_size / 2)
        self.x_max = int(well_x + self.conf.im_size / 2)
        self.y_min = int(well_y - self.conf.im_size / 2)
        self.y_max = int(well_y + self.conf.im_size / 2)
        im_block = im_well[self.y_min:self.y_max, self.x_min:self.x_max]

        if DEBUG:
            cv2.imwrite("./Methods/Methods_saved/im_well_block.png", im_block)
        #cv2.imshow("needle", im_block)
        #cv2.waitKey(0)
        img = np.array(im_block, dtype=np.float32)
        img = np.reshape(img, (1, self.conf.im_size, self.conf.im_size, 1))

        img = img / 255 - 0.5
        self.img = np.reshape(img, (1, self.conf.im_size, self.conf.im_size, 1))

    def blob_tune(self, binary):
        binary = binary * 255
        ret, labels = cv2.connectedComponents(binary)
        blobs_raw = []
        for label in range(1, ret):
            coordinate = np.asarray(np.where(labels == label)).transpose()
            blobs_raw.append(coordinate)
        erosion = cv2.erode(binary, (3, 3), iterations=4)
        ret, labels = cv2.connectedComponents(erosion)
        blobs_tuned = []
        for label in range(1, ret):
            coordinate = np.asarray(np.where(labels == label)).transpose()
            if coordinate.shape[0] > 10:
                blobs_tuned.append(coordinate)
        final_blobs = select_blob(blobs_raw, blobs_tuned)
        tuned_binary = np.zeros(erosion.shape, np.uint8)
        for fblob in final_blobs:
            tuned_binary[fblob[:, 0], fblob[:, 1]] = 1

        return tuned_binary

    def select_big_blobs(self, binary, size = 44):
        binary = binary * 255
        ret, labels = cv2.connectedComponents(binary)
        blobs_tuned = []
        for label in range(1, ret):
            coordinate = np.asarray(np.where(labels == label)).transpose()
            if coordinate.shape[0] > size:
                blobs_tuned.append(coordinate)
        tuned_binary = np.zeros(binary.shape, np.uint8)
        for fblob in blobs_tuned:
            tuned_binary[fblob[:, 0], fblob[:, 1]] = 1

        return tuned_binary, blobs_tuned

    def predict(self, threshold, size = 44):
        pred = self.model.segmen(self.img)

        out_needle = np.zeros((self.conf.ori_size, self.conf.ori_size), np.uint8)
        out_fish = np.zeros((self.conf.ori_size, self.conf.ori_size), np.uint8)

        heatmap_visual = pred[:, :, 0]
        needle_binary = np.zeros(heatmap_visual.shape, np.uint8)
        needle_binary[np.where(heatmap_visual>threshold)] = 1
        if DEBUG:
            cv2.imwrite("./Methods/Methods_saved/needle_binary.png", needle_binary*255)
        #needle_binary = self.blob_tune(needle_binary)
        out_needle[self.y_min:self.y_max, self.x_min:self.x_max] = needle_binary

        #print(needle_binary, needle_binary.shape)
        #cv2.imshow("needle", needle_binary)
        #cv2.waitKey(0)

        heatmap_visual = pred[:, :, 1]
        fish_binary = np.zeros(heatmap_visual.shape, np.uint8)
        fish_binary[np.where(heatmap_visual > threshold)] = 1
        if DEBUG:
            cv2.imwrite("./Methods/Methods_saved/fish_binary.png", fish_binary*255)

        out_fish[self.y_min:self.y_max, self.x_min:self.x_max] = fish_binary

        optimized_binary, fish_blobs = self.select_big_blobs(out_fish, size=size)

        if DEBUG:
            cv2.imwrite("./Methods/Methods_saved/optimized_binary.png", optimized_binary[self.y_min:self.y_max, self.x_min:self.x_max]*255)
        #print(fish_binary, fish_binary.shape)
        #cv2.imshow("fish", out_binary*127)
        #cv2.waitKey(0)
        return out_needle, out_fish, fish_blobs

    def find_needle_point(self, needle_mask):
        """
        :param needle_mask: the binary of the needle heat map: 0/1
        :return: the center of needle point: y, x or (h, w)
        """
        masked = needle_mask*255
        masked_inv = cv2.bitwise_not(masked)
        gray_masked = cv2.bitwise_and(self.ori_im_gray, self.ori_im_gray, mask = masked)
        gray_masked = gray_masked + masked_inv

        threshold = np.min(np.array(gray_masked, dtype=np.int))

        min_index = np.where(gray_masked == threshold)
        cy = (int)(np.round(np.average(min_index[0])))
        cx = (int)(np.round(np.average(min_index[1])))

        #needle_labelled = cv2.circle(self.ori_im, (cx, cy), 3, 255, -1)
        #cv2.imshow("needle", needle_labelled)
        #cv2.waitKey(0)

        return (cy, cx)

    def find_fish_points(self, fish_mask, needle_center, fish_blobs, percentages = []):
        """
        :param fish_mask: the binary of the fish: 0/1
        :param needle_center: the center of the needle: y, x
        :param fish_blobs: the coordinates of the area of the fish
        :param percentages: list of the points to be touched in percentage coordinate system
        :return: list of the coordinates to be touched for the closest fish to the needle
        """
        n_y, n_x = needle_center
        im_skele = self.ori_im.copy()
        distances = []
        fish_centers = []
        for blob in fish_blobs:
            c_y, c_x = np.round(np.average(np.array(blob), axis=0))
            fish_centers.append([c_y, c_x])
            #im_skele = cv2.circle(im_skele, (int(c_x), int(c_y)), 3, 255, -1)
            distances.append((n_x-c_x)**2 + (n_y-c_y)**2)
        closest_ind = np.argmin(np.array(distances))
        closest_blob = fish_blobs[closest_ind]
        closest_center = fish_centers[closest_ind]
        fish_binary = np.zeros(fish_mask.shape, dtype=np.uint8)
        fish_binary[closest_blob[:, 0], closest_blob[:, 1]] = 1

        #moments = cv2.moments(fish_binary*255)
        #cen_x = moments["m10"] / moments["m00"]
        #cen_y = moments["m01"] / moments["m00"]
        #a = moments["m20"] - moments["m00"]*cen_x*cen_x
        #b = 2*moments["m11"] - moments["m00"] * (cen_x**2 + cen_y**2);
        #c = moments["m02"] - moments["m00"]*cen_y*cen_y
        #theta = 0.5 *math.atan((2*moments["m11"]) / (moments["m20"] - moments["m02"]))
        #theta = (theta/math.pi) *180#0 if a==c else math.atan2(b, a-c)/2.0

        skeleton = skeletonize(fish_binary)
        skeleton_cor = np.where(skeleton>0)
        skeleton_cor = np.array([skeleton_cor[0], skeleton_cor[1]]).reshape(2, -1)
        point1 = skeleton_cor[:, 0]
        point2 = skeleton_cor[:, -1]
        #theta = get_angle(point1, point2, closest_center)
        slope, fish_points = get_points(point1, point2, closest_center, percentages)
        im_skele[np.where(skeleton==1)] = [0,255,0]
        cv2.imwrite("skeleton.png", im_skele)
        colors = [[0, 0, 255],
                  [0, 255, 0],
                  [255, 0, 0]]
        for f_p, c in zip(fish_points, colors):
            s_y, s_x = get_starting_point(f_p, needle_center, -1/slope, radius=30)
            im_skele = cv2.line(img=im_skele, pt1=(f_p[1], f_p[0]), pt2 = (s_x, s_y), color=c, thickness=1)
            im_skele = cv2.line(img=im_skele, pt1=(n_x, n_y), pt2=(s_x, s_y), color=c, thickness=1)
            im_skele = cv2.circle(img = im_skele, center = (f_p[1], f_p[0]), radius =1, color = c, thickness = 1)
        #cv2.imshow("fish", im_skele)
        #cv2.waitKey(0)
        return im_skele, fish_points

    def find_fish_point(self, fish_mask, fish_blob, percentages = [0.01]):
        """
        :param fish_mask: the binary of the fish: 0/1
        :param needle_center: the center of the needle: y, x
        :param fish_blobs: the coordinates of the area of the fish
        :param percentages: list of the points to be touched in percentage coordinate system
        :return: list of the coordinates to be touched for the closest fish to the needle
        """

        c_y, c_x = np.round(np.average(np.array(fish_blob), axis=0))

        closest_center = [c_y, c_x]
        fish_binary = np.zeros(fish_mask.shape, dtype=np.uint8)
        fish_binary[fish_blob[:, 0], fish_blob[:, 1]] = 1

        #moments = cv2.moments(fish_binary*255)
        #cen_x = moments["m10"] / moments["m00"]
        #cen_y = moments["m01"] / moments["m00"]
        #a = moments["m20"] - moments["m00"]*cen_x*cen_x
        #b = 2*moments["m11"] - moments["m00"] * (cen_x**2 + cen_y**2);
        #c = moments["m02"] - moments["m00"]*cen_y*cen_y
        #theta = 0.5 *math.atan((2*moments["m11"]) / (moments["m20"] - moments["m02"]))
        #theta = (theta/math.pi) *180#0 if a==c else math.atan2(b, a-c)/2.0

        skeleton = skeletonize(fish_binary)
        skeleton_cor = np.where(skeleton>0)
        skeleton_cor = np.array([skeleton_cor[0], skeleton_cor[1]]).reshape(2, -1)
        point1 = skeleton_cor[:, 0]
        point2 = skeleton_cor[:, -1]
        #theta = get_angle(point1, point2, closest_center)
        slope, fish_points = get_points(point1, point2, closest_center, percentages)
        #cv2.imshow("fish", im_skele)
        #cv2.waitKey(0)
        return fish_points

    def get_keypoint(self, threshold, size_fish):
        out_needle, out_fish, fish_blobs = self.predict(threshold=threshold, size = size_fish)
        needle_y, needle_x = self.find_needle_point(needle_mask = out_needle)
        if len(fish_blobs)>0:
            im_with_points, fish_points = self.find_fish_points(fish_mask=out_fish, needle_center=(needle_y, needle_x),
                                  fish_blobs=fish_blobs, percentages=[0.05, 0.3, 0.7])
        else:
            im_with_points = self.ori_im,
            fish_points = None
        #cv2.imshow("needle", out_needle*255)
        #cv2.imshow("fish", out_fish * 255)
        #cv2.waitKey(0)

        return out_needle, out_fish, im_with_points, fish_points

def get_angle(point1, point2, fish_center):
    y1, x1 = point1
    y2, x2 = point2
    f_y, f_x = fish_center
    distance1 = (f_x - x1)**2 + (f_y - y1)**2
    distance2 = (f_x - x2)**2 + (f_y - y2)**2

    delta_x = x2-x1
    delta_y = y2-y1


    if distance1 < distance2:
        if delta_x < 0:
            theta = math.atan(delta_y / delta_x)
            theta = - (theta / math.pi) * 180
        elif delta_x > 0:
            theta = math.atan(delta_y / delta_x)
            theta = 180 - (theta / math.pi) * 180
        else:
            theta = 90
    else:
        if delta_x > 0:
            theta = math.atan(delta_y / delta_x)
            theta = - (theta / math.pi) * 180
        elif delta_x < 0:
            theta = math.atan(delta_y / delta_x)
            theta = -180 - (theta / math.pi) * 180
        else:
            theta = -90
    #print(point1, point2, fish_center, theta)
    return theta

def get_points(point1, point2, fish_center, percentages):
    y1, x1 = point1
    y2, x2 = point2
    f_y, f_x = fish_center
    distance1 = (f_x - x1)**2 + (f_y - y1)**2
    distance2 = (f_x - x2)**2 + (f_y - y2)**2

    k = (y2-y1) / (x2-x1 + 0.0000001) + 0.0000001

    if distance1 < distance2:
        top_head = point1
        tail_end = point2
    else:
        top_head = point2
        tail_end = point1
    percent_points = []
    for p in percentages:
        p_y = int(np.round((1-p) * top_head[0] + p * tail_end[0]))
        p_x = int(np.round((1-p) * top_head[1] + p * tail_end[1]))
        percent_points.append([p_y, p_x])
    #print(point1, point2, fish_center, theta)
    return k, percent_points

def get_starting_point(key_point, needle_point, k, radius):

    (y0, x0) = key_point
    delta = math.sqrt(radius**2 / (1+k**2))
    x1 = int(np.round(x0 + delta))
    x2 = int(np.round(x0 - delta))
    y1 = int(np.round(k*x1 + y0 - k*x0))
    y2 = int(np.round(k*x2 + y0 - k*x0))

    n_y, n_x = needle_point
    distance1 = (n_x - x1) ** 2 + (n_y - y1) ** 2
    distance2 = (n_x - x2) ** 2 + (n_y - y2) ** 2
    (y, x) = (y1, x1) if distance1<distance2 else (y2, x2)
    return y, x



if __name__ == '__main__':
    time_cnt = time.time()

    unet_test = UNetTest(n_class=2, cropped_size=240, model_path="6000.pth.tar")
    unet_test.load_model()
    im = cv2.imread("dataset/test/Images/100.jpg")
    anno_im = cv2.imread("dataset/test/annotation/100_label.tif")
    unet_test.load_im(im)
    unet_test.get_keypoint(threshold=0.9)
    time_used = time.time() - time_cnt
    print("used time", time_used)
