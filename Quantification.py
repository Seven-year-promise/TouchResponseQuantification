from Methods.UNet_tf.test import *
from Methods.UNet_tf.util import *
from Methods.Tracking import *
import cv2
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt
from Methods.RegionGrowing import Point, RegionGrow
from Methods.QuantificationIndex import QuantificationIndex

COLORS = [[133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234]]

DRAW_FLOW_LINE = False
DRAW_FLOW_POINT = False
SAVE = False
SAVE_VIDEO = False
SHOW = False
SAVE_X_MIN = 100
SAVE_X_MAX = 380
SAVE_Y_MIN = 100
SAVE_Y_MAX = 380

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

def find_next_point(larva_centers, new_blobs):
    new_center_candidates = []
    for b in new_blobs:
        center = np.round(np.average(b, axis=0))
        new_center_candidates.append([int(center[0]), int(center[1])])

    new_centers = list(map(lambda y: min(new_center_candidates, key=lambda x: ((x[0] - y[0])**2 + (x[1] - y[1])**2 )), larva_centers))

    return new_centers

def larva_tracking(video, model_path):
    unet_test = UNetTestTF()
    unet_test.model.load_graph_frozen(model_path=model_path)
    i = 0
    last_binary, this_binary = None, None

    larva_centers = []
    new_video = []

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480, 480))

    for im in video:
        i += 1
        unet_test.load_im(im)
        _, this_binary, larva_blobs = unet_test.predict(threshold=0.9, size=44)

        if last_binary is None:
            for b in larva_blobs:
                center = np.round(np.average(b, axis=0))
                larva_centers.append([int(center[0]), int(center[1])])
        else:
            larva_centers = find_next_point(larva_centers, larva_blobs)
        last_binary = this_binary
        tracked_im = im.copy()
        for ct, cr in zip(larva_centers, COLORS[:len(larva_centers)]):
            tracked_im = cv2.circle(tracked_im, center = (ct[1], ct[0]), radius=2, color=cr, thickness=2)
        cv2.imshow("tracked", tracked_im)
        cv2.waitKey(1)
        new_video.append(tracked_im)
        out.write(tracked_im)
    out.release()

class BehaviorQuantify:
    def __init__(self, im_shape, model_path):
        self.unet_test = UNetTestTF()
        self.unet_test.model.load_graph_frozen(model_path=model_path)

        self.video = None
        self.larva_centers = []
        self.larva_percentage_pointss = []
        self.larva_skeletons = []
        self.im_shape = im_shape

        self.needle_tracker = NeedleTracker()
        self.larva_tracker = LarvaTracker()
        self.larva_tracker2 = ParticleFilter(50)

        self.RG = RegionGrow()

        self.QutifyIndex = QuantificationIndex(n_l_dis_thre=10, move_thre=25)

    def load_video(self, video):
        self.video = video

    def get_skeleton(self, blob):
        fish_binary = np.zeros(self.im_shape, dtype=np.uint8)
        fish_binary[blob[:, 0], blob[:, 1]] = 1
        skeleton = skeletonize(fish_binary)
        skeleton_cor = np.where(skeleton > 0)
        skeleton_cor = np.array([skeleton_cor[0], skeleton_cor[1]]).reshape(2, -1)
        point1 = skeleton_cor[:, 0]
        point2 = skeleton_cor[:, -1]

        return skeleton_cor

    def quantification_init(self):
        self.larva_centers = []
        self.larva_percentage_pointss = []
        self.larva_skeletons = []
        if self.video is None:
            print("please load the video first")
        else:
            self.unet_test.load_im(self.video[0])
            needle_binary, _, larva_binary, larva_blobs = self.unet_test.predict(threshold=0.9, size=12)

            cv2.imwrite("tracking_saved/original_im.jpg", self.video[0][SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
            cv2.imwrite("tracking_saved/larva_binary.jpg", larva_binary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX]*255)
            cv2.imwrite("tracking_saved/needle_binary.jpg", needle_binary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX]*255)
            needle_point = self.unet_test.find_needle_point(needle_binary)
            larva_points = []
            for b in larva_blobs:
                center = np.round(np.average(b, axis=0))
                fish_point = self.unet_test.find_fish_point(fish_mask=larva_binary, fish_blob=b, percentages=[0.1])
                #print(fish_point)
                self.larva_centers.append(center)
                self.larva_percentage_pointss.append(fish_point)
                skel = self.get_skeleton(b)

                self.larva_skeletons.append(skel)

            self.needle_tracker.init_p0(needle_point)
            self.larva_tracker.init_p0(self.larva_percentage_pointss)
            first_gray = self.preprocessing(self.video[0])
            self.larva_tracker2.init_boxes0(first_gray, self.larva_centers, larva_blobs)

    def preprocessing(self, im, strong = False):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if not strong:
            _, (well_x, well_y, _), im_well = well_detection(im, gray)
        else:
            _, (well_x, well_y, _), im_well = well_detection_strong(im, gray, threshold=150)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

        return im_well

    def larva_touched(self, needle_point, n_l_dis_thre):
        """
        decide which larva was touched
        :param needle_point: the needle point of the last frame
        :param n_l_dis_thre: the distance to decide that the larva is touched
        :return: 0, 1, 2, 3: the larva in dex touched, -1: none of the larvae touched
        """
        distances = []
        for c in self.larva_centers:
            d = math.sqrt((c[0] - needle_point[0])**2 + (c[1] - needle_point[1])**2)
            distances.append(d)
        touched_ind = np.argmin(np.array(distances))
        print(distances)
        if distances[touched_ind] < n_l_dis_thre:
            return touched_ind
        else:
            return -1

    def compute_total_distances(self, all_points, ind):
        distance = 0
        #print(all_points)
        for i in range(len(all_points)-1):
            p1 = all_points[i][ind]
            p2 = all_points[i+1][ind]
            d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            distance += d
            #print(p1, p2, d, distance)
        #print(distance)
        return distance

    def local_seg(self, frame, larva_points):
        """
        segmentation for the local area of the larva
        :param frame: frame to generate the local area of the larva
        :param larva_points: the particles output from the tracking procedure
        :return:
            tuned_binary: the optimized binary in original image size
            larva_patches: the larva areas in local size
            fblobs: the larva blobs coordinates in original image size
            tuned_points: the optimized larva centers
        """
        # the fish area should be enlarged to 40 by 40
        fblobs = []
        larva_patches = []
        tuned_points = []
        tuned_binary = frame.copy() #np.zeros(frame.shape, dtype='uint8') #
        tuned_binary[:, :] = 0
        for l_p in larva_points:
            y_ave = int(l_p[0])
            x_ave = int(l_p[1])

            blur = cv2.medianBlur(frame, 5)
            binary = self.RG.regionGrowLocalApply(blur,
                                                  [Point(x_ave, y_ave)], # Point (x, y)
                                                  diff_thre=20,
                                                  binary_high_thre=220,
                                                  binary_low_thre=50,
                                                  size_thre=200)
            #blur = cv2.GaussianBlur(im, (3, 3), 0)
            #local_coors = np.asarray(np.where(binary == 1)).transpose()
            #x_min =
            #ret, th = cv2.threshold(larva_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            #ret, th = cv2.threshold(larva_patch, 220, 255, cv2.THRESH_BINARY)
            #kernel = np.ones((3, 3), dtype=np.uint8)
            #closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            #th = closing
            #binary = np.zeros(th.shape, np.uint8)
            #binary[np.where(th == 0)] = 1
            #binary[np.where(th == 255)] = 0


            ret, labels = cv2.connectedComponents(binary)
            blobs_raw = []
            size = []

            for label in range(1, ret):
                coordinate = np.asarray(np.where(labels == label)).transpose()
                size.append(coordinate.shape[0])
                blobs_raw.append(coordinate)
                tuned_y_ave = np.average(coordinate[:, 0])
                tuned_x_ave = np.average(coordinate[:, 1])
                tuned_points.append([tuned_y_ave, tuned_x_ave])

            biggest_ind = np.argmax(np.array(size))
            blob = blobs_raw[biggest_ind]
            # get the small singe larva patch
            x_min = np.min(blob[:, 1])
            y_min = np.min(blob[:, 0])
            x_max = np.max(blob[:, 1])
            y_max = np.max(blob[:, 0])
            larva_patch = np.zeros(((y_max-y_min+1), (x_max-x_min+1)), np.uint8)
            larva_patch[blob[:, 0] - y_min, blob[:, 1] - x_min] = 1
            larva_patches.append(larva_patch)
            #cv2.imshow("binary", larva_patch + 255)
            #cv2.waitKey(0)
            # remap the flob to the original size

            fblobs.append(blob)
            tuned_binary[blob[:, 0], blob[:, 1]] = 1

        #cv2.imshow("local", tuned_binary*255)
        #cv2.waitKey(0)

        return tuned_binary, larva_patches, fblobs, tuned_points


    def quantify(self, save_path, video_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        new_video = []
        new_video2 = []
        new_video3 = []
        new_video4 = []
        new_video5 = []
        moving_positions = []
        radiuses = []
        response_begin_time = 0
        Response_time = 0
        old_im = self.video[0]
        old_gray = self.preprocessing(self.video[0])
        needle_points = []
        larva_pointss = []
        larva_patchess = []
        larva_blobss = []
        num_diffss = []

        id_im = 0
        previous = []
        previous.append(old_gray)
        for im in self.video[1:]:
            id_im += 1
            new_gray = self.preprocessing(im, strong=True)
            #cv2.imshow("well", new_gray)
            #cv2.waitKey(1)
            im_with_pars = im.copy()
            draw_particles(im_with_pars, self.larva_tracker2.new_particles)
            needle_point = self.needle_tracker.track(old_gray, new_gray)
            needle_points.append(needle_point)
            #larva_points = self.larva_tracker.optical_track(old_gray, new_gray)
            #tracked_im = self.larva_tracker.dense_track(old_im, im)
            #larva_pointss.append(larva_points)
            larva_points, im_diff, num_diffs = self.larva_tracker2.track(previous, new_gray, 15, 0.5)
            tracked_binary, tracked_patches, tracked_blobs, tracked_points = self.local_seg(new_gray, larva_points)
            #self.larva_tracker2.resampling_within_blobs(tracked_blobs)
            larva_pointss.append(tracked_points)
            larva_patchess.append(tracked_patches)
            larva_blobss.append(tracked_blobs)
            num_diffss.append(num_diffs)
            """
            modify the larva area according to the rough tracking
            """

            #_ = self.larva_tracker.dense_track(old_im, im)
            #_ = self.larva_tracker.difference_track(previous, new_gray, 10)
            tracked_im = im.copy()
            tracked_im2 = im.copy()
            #for ct, cr in zip(larva_centers, COLORS[:len(larv<a_centers)]):

            if DRAW_FLOW_LINE:
                show_n_points = needle_points[::20]

                for i in range(len(show_n_points) - 1):
                    '''
                    tracked_im = cv2.line(tracked_im, pt1=(show_n_points[i][1], show_n_points[i][0]),
                                          pt2=(show_n_points[i+1][1], show_n_points[i+1][0]),
                                          color=(0, 255, 0), thickness=2)
                    '''
                    tracked_im = cv2.line(tracked_im, pt1=(show_n_points[i][1], show_n_points[i][0]),
                                          pt2=(show_n_points[i + 1][1], show_n_points[i + 1][0]),
                                          color=(0, 255, 0), thickness=3)
                show_l_points = larva_pointss[::20]
                for i in range(len(show_l_points) - 1):
                    if len(show_l_points[i]) == len(show_l_points[i+1]):
                        for l_p1, l_p2, c in zip(show_l_points[i], show_l_points[i+1], COLORS[:len(show_l_points[i])]):
                            # print(l_p)
                            '''
                            tracked_im = cv2.line(tracked_im, pt1=(int(np.round(l_p1[0])), int(np.round(l_p1[1]))),
                                                  pt2=(int(np.round(l_p2[0])), int(np.round(l_p2[1]))),
                                                  color=c, thickness=2)
                            '''
                            tracked_im = cv2.line(tracked_im, pt1=(int(np.round(l_p1[1])), int(np.round(l_p1[0]))),
                                                  pt2=(int(np.round(l_p2[1])), int(np.round(l_p2[0]))),
                                                  color=c, thickness=3)
            if DRAW_FLOW_POINT:
                tracked_im2 = cv2.rectangle(tracked_im2, pt1=(needle_point[1], needle_point[0]),
                                          pt2=(needle_point[1]+3, needle_point[0]+3),
                                          color=(0, 255, 0), thickness=4)

                for l_p1, c in zip(larva_points, COLORS[:len(larva_points)]):
                    # print(l_p)
                    tracked_im2 = cv2.rectangle(tracked_im2, pt1=(int(np.round(l_p1[1])), int(np.round(l_p1[0]))),
                                          pt2=(int(np.round(l_p1[1]+3)), int(np.round(l_p1[0]+3))),
                                          color=c, thickness=4)

            if SAVE:
                if not os.path.exists(save_path + "/" + video_name):
                    os.makedirs(save_path + "/" + video_name)
                cv2.imwrite(save_path + "/" + video_name + "/ori" + str(id_im) + ".jpg", im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(save_path + "/" + video_name + "/particles_line" + str(id_im) + ".jpg", tracked_im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(save_path + "/" + video_name + "/particles_point" + str(id_im) + ".jpg", tracked_im2[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(save_path + "/" + video_name + "/particles_ori" + str(id_im) + ".jpg", im_with_pars[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(save_path + "/" + video_name + "/particles_difference" + str(id_im) + ".jpg", im_diff[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                local_bianary = tracked_binary*255
                cv2.imwrite(save_path + "/" + video_name + "/local_seg" + str(id_im) + ".jpg", local_bianary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                print("saving pictures")
            if SHOW:
                cv2.imshow("local seg", tracked_binary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX]*255)
                cv2.imshow("tracked", tracked_im2[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imshow('new_gray', im_with_pars[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imshow('diff_im', im_diff[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.waitKey(1)
            old_gray = new_gray
            old_im = im
            previous.append(old_gray)
            #print(str(id_im))
            new_video3.append(im_with_pars[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
            new_video4.append(im_diff[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
            new_video5.append(tracked_binary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX]*255)
            #print(larva_points[1])

        larva_touched = self.larva_touched(needle_points[-1], 10)


        #print(larva_pointss)
        if SAVE_VIDEO:
            # decide which larva is the one to touch

            distances = []
            for count, im in enumerate(self.video[1:]):
                show_n_points = needle_points[:count+1:20]
                tracked_im = im.copy()
                for i in range(len(show_n_points) - 1):
                    tracked_im = cv2.line(tracked_im, pt1=(show_n_points[i][1], show_n_points[i][0]),
                                          pt2=(show_n_points[i + 1][1], show_n_points[i + 1][0]),
                                          color=(0, 255, 0), thickness=2)
                show_l_points = larva_pointss[:count+1:20]
                for i in range(len(show_l_points) - 1):
                    l_p1, l_p2 = show_l_points[i][larva_touched], show_l_points[i + 1][larva_touched]
                    tracked_im = cv2.line(tracked_im, pt1=(int(np.round(l_p1[1])), int(np.round(l_p1[0]))),
                                          pt2=(int(np.round(l_p2[1])), int(np.round(l_p2[0]))),
                                          color=COLORS[2], thickness=2)
                new_video.append(tracked_im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                new_video2.append(tracked_im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                #cv2.imshow('tracked_im', tracked_im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                if count >1:
                    points_for_distance = larva_pointss[0:count+1]
                    #print(points_for_distance)
                    #cv2.waitKey(0)
                    distances.append(self.compute_total_distances(points_for_distance, larva_touched))

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            #out = cv2.VideoWriter(save_path + video_name + 'line.avi', fourcc, 20.0, (SAVE_Y_MAX-SAVE_Y_MIN+1, SAVE_X_MAX-SAVE_X_MIN+1))
            out2 = cv2.VideoWriter(save_path + video_name + 'point.avi', fourcc, 20.0, (SAVE_Y_MAX-SAVE_Y_MIN+1, SAVE_X_MAX-SAVE_X_MIN+1))
            out3 = cv2.VideoWriter(save_path + video_name + 'particles.avi', fourcc, 20.0, (SAVE_Y_MAX-SAVE_Y_MIN+1, SAVE_X_MAX-SAVE_X_MIN+1))
            #out4 = cv2.VideoWriter('tracking_saved/output4.avi', fourcc, 20.0, (SAVE_Y_MAX-SAVE_Y_MIN+1, SAVE_X_MAX-SAVE_X_MIN+1))
            out5 = cv2.VideoWriter(save_path + video_name + 'rg.mp4', mp4, 20.0,
                                   (SAVE_Y_MAX - SAVE_Y_MIN + 1, SAVE_X_MAX - SAVE_X_MIN + 1), False)
            for im1, im2, im3, im4, im5 in zip(new_video[::10], new_video2[::10], new_video3[::10], new_video4[::10], new_video5[::10]):
                #out.write(im1)
                out2.write(im2)
                out3.write(im3)
                #out4.write(im4)
                out5.write(im5)

            t = np.arange(len(distances))
            plt.plot(t, distances)
            plt.xlabel("t (ms)")
            plt.ylabel("distance (pixels)")
            plt.title("The Distance that the Larva Moved")
            plt.savefig(save_path + video_name + 'moving_distance.png',)
            plt.close()
            print("saving videos")
            #out.release()
            out2.release()
            out3.release()
            #out4.release()
            out5.release()

        # save immediate data

        #num_diffss_df = pd.DataFrame(num_diffss)

        #num_diffss_df.to_csv(save_path + "num_diffss.csv", index=False, header=False)

        # get the quantification indexes

        if larva_touched < 0:
            print("None touched")
            return -2, -2, -2, -2, -2 #None, None, None, None
        else:
            t_l, c_m, cpt, t_r, d_m = self.QutifyIndex.get_indexes(larva_first_centers=self.larva_centers,
                                                               needle_points=needle_points,
                                                               larva_pointss=larva_pointss,
                                                               larva_patchess=larva_patchess,
                                                               num_diffss=num_diffss,
                                                               larva_touched=larva_touched)

            print(t_l, c_m, cpt, t_r, d_m)
            return t_l, c_m, cpt, t_r, d_m

if __name__ == '__main__':
    behav_quantify = BehaviorQuantify((480, 480), model_path="./Methods/UNet_tf/ori_UNet/models-trained-on200-2/models_rotation_contrast/UNet30000.pb")
    base_path = "./Methods/Multi-fish_experiments/"
    date = ["20210507-1_73hpf_veli/"] #["20210522-4compounds/"]#["20210414/", "20210415-1/", "20210415-2/", "20210416-1/", "20210416-2/"]
    capacity = ["6white/"]# ["1control/", "2blue/", "3green/", "4yellow/", "5red/",  ["Caffine/", "Saha/"] #"Control/", "Dia/", "DMSO/", "Iso/",
    touching_part = [""]
    save_path = "./tracking_saved/"
    quantification_result_path = "./QuantificationResults/"
    for d in date:
        for c in capacity:
            for p in touching_part:
                this_path = base_path + d + c + p
                file_names = [f for f in os.listdir(this_path) if f.endswith('.avi')]
                print(file_names)
                video_cnt = 0

                # for saving the quantification
                result_path = quantification_result_path + d + c + p
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                result_csv_file = result_path + "quantification.csv"
                result_csv_file = open(result_csv_file, "w", newline="")
                result_csv_writer = csv.writer(result_csv_file, delimiter=",")
                result_csv_writer.writerow(["video_name", "t_l", "c_m", "cpt", "t_r", "d_m"])

                for f in file_names:
                    if f[-3:] == "avi":
                        video_cnt += 1
                        print("NO.", video_cnt, this_path + f)
                        video_path = this_path + f
                        #f = "WT_114643_Speed25.avi"
                        #video_path = "./Methods/Multi-fish_experiments/20210522-4compounds/Iso/WT_114643_Speed25.avi"
                        video = []
                        cap = cv2.VideoCapture(video_path)
                        success, frame = cap.read()
                        while success:
                            video.append(frame)
                            success, frame = cap.read()
                        cap.release()

                        behav_quantify.load_video(video)
                        behav_quantify.quantification_init()
                        t_l, c_m, cpt, t_r, d_m = behav_quantify.quantify(save_path = save_path+d + c + p, video_name=f)

                        # for saving the quantification
                        result_csv_writer.writerow([f, t_l, c_m, cpt, t_r, d_m])

                        #cv2.waitKey(0)
                        #larva_tracking(video[3000:4000], model_path="./Methods/UNet_tf/LightCNN/models_rotate_contrast/UNet60000.pb")
                    #if video_cnt > 0:
                        #break

                # for saving the quantification
                result_csv_file.close()

    cv2.destroyAllWindows()