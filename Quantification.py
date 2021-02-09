from Methods.UNet_tf.test import *
from Methods.UNet_tf.util import *
from Methods.Tracking import *
import cv2
import numpy as np

COLORS = [[133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234]]

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
        self.larva_skeletons = []
        self.im_shape = im_shape

        self.needle_tracker = NeedleTracker()
        self.larva_tracker = LarvaTracker()

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
        if self.video is None:
            print("please load the video first")
        else:
            self.unet_test.load_im(self.video[0])
            needle_binary, larva_binary, larva_blobs = self.unet_test.predict(threshold=0.9, size=44)
            needle_point = self.unet_test.find_needle_point(needle_binary)
            for b in larva_blobs:
                center = np.round(np.average(b, axis=0))
                self.larva_centers.append([int(center[0]), int(center[1])])
                skel = self.get_skeleton(b)

                self.larva_skeletons.append(skel)
            self.needle_tracker.init_p0(needle_point)

    def preprocessing(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, (well_x, well_y, _), im_well = well_detection(im, gray)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

        return im_well

    def quantify(self):
        moving_positions = []
        radiuses = []
        response_begin_time = 0
        Response_time = 0
        old_gray = self.preprocessing(self.video[0])
        for im in self.video[1:]:
            new_gray = self.preprocessing(im)

            needle_point = self.needle_tracker.track(old_gray, new_gray)

            larva_points = self.larva_tracker.track()

if __name__ == '__main__':
    video_path = "./Methods/Multi-fish_experiments/20200121/5/body/WT_153930_Speed25.avi"
    video = []
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    while success:
        video.append(frame)
        success, frame = cap.read()
    cap.release()
    larva_tracking(video[3000:4000], model_path="./Methods/UNet_tf/LightCNN/models_rotate_contrast/UNet60000.pb")
    cv2.destroyAllWindows()