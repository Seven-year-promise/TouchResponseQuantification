import cv2
import numpy as np



def optical_flow(old_gray, new_gray, p0, lk_params):
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    # draw the tracks

    #cv2.imshow('frame', img)
    return good_new

class NeedleTracker(im):
    def __init__(self, init_point):
        feature_params = dict(maxCorners=20,
                              qualityLevel=0.8,
                              minDistance=7,
                              blockSize=50)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0 = np.array([[init_point[1], init_point[0]]])

    def track(self, old_gray, new_gray):
        good_new = optical_flow(old_gray, new_gray, self.p0, self.lk_params)
        if len(good_new) == 0:
            good_new = self.p0[0]
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
        (y_offset, x_offset) = self.needle_usingMaxima(new_gray[(new_y - 7):(new_y + 7), (new_x - 7):(new_x + 7)])
        new_x = new_x - 7 + x_offset
        new_y = new_y - 7 + y_offset
        good_new[0][0] = new_x
        good_new[0][1] = new_y
        self.p0 = good_new.reshape(-1, 1, 2)
        return good_new

    def needle_usingMaxima(self, gray, blur='false'):
        """
        given an image, find the lowest pixel
        return: h, w
        """
        if blur:
            gray = cv2.medianBlur(gray, 3)
        threshold = np.min(np.array(gray, dtype=np.int))
        min_index = np.where(gray == threshold)
        cy = (int)(np.round(np.average(min_index[0])))
        cx = (int)(np.round(np.average(min_index[1])))
        return (cy, cx)

class LarvaTracker:
    def __init__(self):
        self.po = [0, 0]


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
