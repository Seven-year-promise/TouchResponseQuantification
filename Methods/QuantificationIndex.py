import cv2
import numpy as np
import os
import argparse
import time

from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert

from detection import needle_usingMaxima, circle_detection, object_detection, fish_keypoints, maxima_blob_counter
from Curvature import ComputeCurvature

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--touching_part', type=str, default='tail',
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
touching_index = 0
video_path = ''
if args.touching_part == 'head':
    video_path = "./videos/head/all/"
    touching_index = 0
elif args.touching_part == 'body':
    video_path = "./videos/body/all/"
    touching_index = 1
elif args.touching_part == 'tail':
    video_path = "./videos/tail/all/"
    touching_index = 2
else:
    print("please select the correct part! ")


def preprocessing(frame, well_info, well_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    # (cx, cy) = ((int)(np.round(frame.shape[1] / 2)), (int)(np.round(frame.shape[0] / 2)))
    cv2.circle(mask, (well_info[0], well_info[1]), well_info[2], 255, -1)

    masked = cv2.bitwise_and(gray, gray, mask=well_mask)

    return masked


def optical_flow(old_gray, new_gray, p0, lk_params):
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    # draw the tracks

    # cv2.imshow('frame', img)
    return good_new


def frame_difference(old_frame, new_frame, motion_threshold):
    old_frame = np.array(old_frame, dtype=np.int)
    new_frame = np.array(new_frame, dtype=np.int)

    difference = np.abs(old_frame - new_frame)
    difference = np.array(difference, dtype='uint8')

    ret, binary = cv2.threshold(difference, 10, 255, cv2.THRESH_BINARY)
    median = cv2.medianBlur(binary, 3)
    # cv2.imshow('difference', median)
    # cv2.waitKey(1)

    # blobs, centers = blobDetection(binary, 20, size_threshold_High = 200, size_threshold_Low = 20)

    changedIndex = np.where(median > 0)

    # print(len(changedIndex[0]))
    positive = median[changedIndex]

    if len(changedIndex[0]) > motion_threshold:
        moving_centerY = np.average(changedIndex[0])
        moving_centerX = np.average(changedIndex[1])
        # if not os.path.exists("difference.png"):
        """
        old_frame = old_frame[((int)(moving_centerY -30)):((int)(moving_centerY +30)), ((int)(moving_centerX -30)):((int)(moving_centerX +30))]
        new_frame = new_frame[((int)(moving_centerY -30)):((int)(moving_centerY +30)), ((int)(moving_centerX -30)):((int)(moving_centerX +30))]
        median = median[((int)(moving_centerY -30)):((int)(moving_centerY +30)), ((int)(moving_centerX -30)):((int)(moving_centerX +30))]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 3),
                                     sharex=True, sharey=True)

        ax = axes.ravel()

        ax[0].imshow(old_frame, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Previous frame', fontsize=10)

        ax[1].imshow(new_frame, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Current frame', fontsize=10)

        ax[2].imshow(median, cmap=plt.cm.gray)
        ax[2].axis('off')
        ax[2].set_title('Difference', fontsize=10)
        plt.show()
        """
        return True, len(changedIndex[0]), [moving_centerY, moving_centerX]
    else:
        return False, len(changedIndex[0]), None


def motion_detection(old_frame, new_frame, motion_threshold=15):
    # flag, difference = frame_difference(old_frame, new_frame)
    # print(difference)
    return frame_difference(old_frame, new_frame, motion_threshold)


def Angle_detection(img, fish_box):
    """
    the fish size: height 40
                   width 12
    the box of fish is 40 X 40
                x_min, y_min, x_max, y_max
    """
    fish_area = img[fish_box[1]:fish_box[3], fish_box[0]:fish_box[2], :]
    """
    #cv2.imshow('fish', fish_area)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    f_height = fish_area.shape[0]
    f_width = fish_area.shape[1]
    X = np.ones((f_height, f_width), dtype=np.int)
    Y = np.ones((f_height, f_width), dtype=np.int)
    #cv2.imshow('fish', fish_area)
    #cv2.waitKey(10)
    for i in range(f_height):
        for j in range(f_width):
            X[i, j] = j
            Y[i, j] = i
    #ax.plot_surface(X = X, Y = Y, Z = fish_area)
    """
    return find_angle(fish_area)
    # plt.show()


def find_angle(fish_part):
    """
    the fish size: height 40
                   width 12
    the box of fish is 40 X 40
                x_min, y_min, x_max, y_max
    """
    (height, width) = fish_part.shape[:2]
    # fish_part = cv2.medianBlur(fish_part, 3)
    fish_part = cv2.cvtColor(fish_part, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(fish_part, 150, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(threshold)
    # median = cv2.medianBlur(binary, 3)
    """
    (h, w) = median.shape[:2]
    mask = np.ones((h+2, w+2), dtype = np.uint8)
    median = cv2.floodFill(median, mask, (0,0), 255)
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    blob = maxima_blob_counter(closing)

    skeleton = skeletonize(blob)

    skeleton_cor = np.where(skeleton > 0)
    (skeleton_y, skeleton_x) = (skeleton_cor[0], skeleton_cor[1])
    if len(skeleton_x) < 4:
        return None
    else:
        skeleton_minx = np.min(skeleton_x)
        skeleton_miny = np.min(skeleton_y)
        skeleton_maxx = np.max(skeleton_x)
        skeleton_maxy = np.max(skeleton_y)
        ComputeCur = ComputeCurvature(degree=3)
        radius = ComputeCur.non_linear_fit(skeleton_x, skeleton_y)
        return radius
    """
    print('cur', para)
    # display results

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 4),
                             sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(fish_part, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(closing, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('binary', fontsize=20)

    ax[2].imshow(skeleton, cmap=plt.cm.gray)
    ax[2].axis('off')
    ax[2].set_title('skeleton', fontsize=20)

    plotx = np.arange(skeleton_minx-1, skeleton_maxx+2)
    ploty = ComputeCur.func(para, plotx)
    ax[0].plot(plotx, ploty, color='green', linewidth=3)

    ax[3].scatter(x, y, color='black')
    ax[3].plot(plotx, ploty, color='blue', linewidth=3)
    #ax[3].xticks(())
    #ax[3].yticks(())
    ax[3].axis('off')
    ax[3].set_title('cur_fitting', fontsize=20)

    fig.tight_layout()
    #plt.show()


    #plt.plot(x, y_pred, color='m')
    #plt.show()
    plt.scatter(x, y, color='black')
    plt.plot(plotx, ploty, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    """
    """
    size = np.size(fish_part)
    skel = np.zeros(fish_part.shape,np.uint8)

    ret,fish_part = cv2.threshold(fish_part,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(fish_part,element)

        temp = cv2.dilate(eroded,element)
        cv2.imshow("skel",temp)
        cv2.waitKey(0)
        temp = cv2.subtract(fish_part,temp)
        skel = cv2.bitwise_or(skel,temp)

        fish_part = eroded.copy()

        zeros = size - cv2.countNonZero(fish_part)
        if zeros==size:
            done = True

    cv2.imshow("skel",skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """



def Tracking(success,
             well_info,
             well_mask,
             cap,
             first_frame,
             needle_point,
             skeleton_cor,
             motion_threshold):
    """
    Assumption: the fish does not move before touched.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("\t framerate:", fps, motion_threshold)

    old_gray = preprocessing(first_frame, well_info, well_mask)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=20,
                          qualityLevel=0.8,
                          minDistance=7,
                          blockSize=50)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(100, 255, (250, 3))
    old_ones = []
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    p0[0, 0, 0] = needle_point[1]
    p0[0, 0, 1] = needle_point[0]
    p0 = np.array([p0[0, :, :]])
    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)
    motion_differences = []
    frame_cnt = 0
    (touchingx, touchingy) = (skeleton_cor[:, 1], skeleton_cor[:, 0])
    skeleton_num = len(touchingx)
    Latency_time = 0
    Latency_time_finished = False
    touching_flag = False
    move_flag = False
    saving_flag = False

    final_position = [0, 0]
    position_flag = False
    final_img = first_frame.copy()
    im_fish_trajectory = first_frame.copy()
    stop_cnt = 0
    moving_positions = []
    radiuses = []
    response_begin_time = 0
    Response_time = 0
    while success:
        success, frame = cap.read()
        TIME = time.clock()
        if not success:
            break
        frame_gray = preprocessing(frame, well_info, well_mask)

        good_new = optical_flow(old_gray, frame_gray, p0, lk_params)
        if len(good_new) == 0:
            good_new = p0[0]
        (new_x, new_y) = (((int))(np.round(good_new[0][0])), (int)(np.round(good_new[0][1])))
        # cv2.imshow("needle", frame_gray)
        # cv2.waitKey(1)
        if new_x < 7:
            new_x = 7
        if new_y < 7:
            new_y = 7
        if new_x > 473:
            new_x = 473
        if new_y > 473:
            new_y = 473
        (y_offset, x_offset) = needle_usingMaxima(frame_gray[(new_y - 7):(new_y + 7), (new_x - 7):(new_x + 7)])
        new_x = new_x - 7 + x_offset
        new_y = new_y - 7 + y_offset
        good_new[0][0] = new_x
        good_new[0][1] = new_y
        old_ones.append([new_x, new_y])

        distance_to_touching = (new_x - touchingx) * (new_x - touchingx) + (new_y - touchingy) * (new_y - touchingy)
        # print("distance is:", distance_to_touching)
        close_distances = []
        for n in range(skeleton_num):
            # frame = cv2.circle(frame, (touchingx[n], touchingy[n]),  1, (255, 255, 0), thickness = 1)
            close_distances.append(
                (new_x - touchingx[n]) * (new_x - touchingx[n]) + (new_y - touchingy[n]) * (new_y - touchingy[n]))
        distance_to_touching = np.min(close_distances)
        distance_index = np.argmin(close_distances)
        if distance_to_touching < 6 * 6:
            # print(distance_to_touching)
            touching_flag = True
            # frame = cv2.circle(frame, (new_x, new_y),  2, (0, 255, 255))
            # if not saving_flag:
            # frame = cv2.circle(frame, (touchingx[distance_index], touchingy[distance_index]),  1, (0, 255, 255), thickness = 2)

            # path = './closetofish/' + str(TIME) + '.png'
            # print(path)
            # cv2.imwrite('./closetofish/' + str(TIME) + '.png', frame)
            # saving_flag = True
            # cv2.imshow("touch", frame)
            # cv2.waitKey(0)

        if not touching_flag:
            Latency_time = frame_cnt

        move_flag, motion_difference, moving_position = motion_detection(old_gray, frame_gray, motion_threshold)

        if move_flag and touching_flag and not Latency_time_finished:
            Latency_time_finished = True
            print("\t latency from the frame ", Latency_time, "to frame ", frame_cnt, Latency_time_finished)
            print("\t Moving pixels ", motion_difference, motion_threshold)
            Latency_time = frame_cnt - Latency_time
            response_begin_time = frame_cnt

        # if move_flag:
        # print("movement detected from frame:", frame_cnt, move_flag)
        # cv2.imwrite("./images/" + str(frame_cnt) + ".png", frame_gray)
        # plt.ylabel("frame" + str(frame_cnt))
        # plt.show()

        """
        if move_flag:
            fish_box = [(int)(moving_position[1] - 20),  (int)(moving_position[0] - 20), (int)(moving_position[1] + 20),  (int)(moving_position[0] + 20)]
            print("moving_position", fish_box)
            fish_part = Angle_detection(frame_gray, fish_box)
            cv2.imwrite('./movement_crops/' + str(frame_cnt) + '.png', fish_part)
        """

        if Latency_time_finished and move_flag:
            final_position = moving_position
            moving_positions.append(moving_position)
            fish_box = [(int)(moving_position[1] - 20), (int)(moving_position[0] - 20), (int)(moving_position[1] + 20),
                        (int)(moving_position[0] + 20)]
            radius = Angle_detection(frame, fish_box)
            if radius is not None:
                radiuses.append(radius)
            Response_time = frame_cnt - response_begin_time
            print("\t response from the frame ", response_begin_time, "to frame ", frame_cnt)

        motion_differences.append(motion_difference)

        # for old_one in old_ones:
        # final_img = cv2.circle(frame, (old_one[0], old_one[1]),  1, (0, 255, 0))

        # cv2.imshow("img", final_img)

        k = 0xff  # cv2.waitKey(1) &
        if k == 'q':
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        im_fish_trajectory = frame.copy()
        p0 = good_new.reshape(-1, 1, 2)
        frame_cnt += 1
    """
    if not move_flag and not Latency_time_finished:
        print("\t from the frame ", Latency_time, "to frame ", frame_cnt)
        Latency_time = frame_cnt - Latency_time
    """
    # for old_one in old_ones:
    # final_img = cv2.circle(frame, (old_one[0], old_one[1]),  1, (0, 255, 0))
    # cv2.imwrite('./results/tracking_result/' + str(TIME) + '.png', final_img)

    # cv2.imshow("final_img.png", final_img)
    x_axis = np.arange(frame_cnt)
    Latency_time = Latency_time * 1.0 / fps
    Response_time = Response_time * 1.0 / fps
    # print("\t Latency time:", Latency_time)
    # print("\t Response time:", Response_time)
    # plt.plot(motion_differences)
    # plt.ylabel("number of positive pixels")
    # plt.xlabel('Key Frame')
    # plt.show()
    # cv2.waitKey(0)
    not_moving = 0
    if Latency_time_finished:
        return Latency_time, Response_time, final_position, moving_positions, radiuses, not_moving
    else:
        if touching_flag:
            not_moving = 1
        return None, None, None, None, None, not_moving
