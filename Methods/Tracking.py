import cv2
import numpy as np

from scipy import signal
from pylab import *
import cv2
import random
from Methods.ParticleFilter import create_Gaussian_particles
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

Epsilon = 1e-6

def optical_flow(old_gray, new_gray, p0, lk_params):
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    # draw the tracks

    #cv2.imshow('frame', img)
    return good_new

def optical_flow_pixel_ori(Ix, Iy, It, pt, kernel_size):
    IX = []
    IY = []
    IT = []
    for h in range(int(pt[0]) - kernel_size//2, int(pt[0]) + kernel_size//2 + 1):
        for w in range(int(pt[1]) - kernel_size // 2, int(pt[1]) + kernel_size // 2 + 1):
            IX.append(Ix[h, w])
            IY.append(Iy[h, w]) # Ix, Iy
            IT.append(It[h, w])  # It


    # Using the minimum least squares solution approach
    LK = (IX, IY)
    LK = np.matrix(LK)
    LK_T = np.array(np.matrix(LK))  # transpose of A
    LK = np.array(np.matrix.transpose(LK))

    A1 = np.dot(LK_T, LK)  # Psedudo Inverse
    A2 = np.linalg.pinv(A1)
    A3 = np.dot(A2, LK_T)

    u, v = np.dot(A3, IT)  # we have the vectors with minimized square error

    """
    
    A = np.array(A, np.float32). reshape((kernel_size ** 2, 2))
    b = np.array(b, np.float32).reshape((kernel_size ** 2, 1))

    AT = A.transpose()
    ATA = np.matmul(AT, A)
    ATb = np.matmul(AT, b)

    ATA_inv = np.linalg.pinv(ATA)
    d = -1 * np.matmul(ATA_inv, ATb)
    d[d>1] = 0
    """
    return u, v #d[0, 0], d[1, 0]

def optical_flow_ori(old_gray, new_gray, p0, kernel_size):
    pt = [p0[0, 0, 0], p0[0, 0, 1]]
    I1_smooth = cv2.GaussianBlur(old_gray  # input image
                                 , (3, 3)  # shape of the kernel
                                 , 0  # lambda
                                 )
    I2_smooth = cv2.GaussianBlur(new_gray, (3, 3), 0)
    Ix = signal.convolve2d(I1_smooth, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, 0.25],
                                                                                                   [-0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in Y direction
    Iy = signal.convolve2d(I1_smooth, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, -0.25],
                                                                                                   [0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in XY direction
    It = signal.convolve2d(I1_smooth, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                [[-0.25, -0.25],
                                                                                                 [-0.25, -0.25]],
                                                                                                'same')
    delta_x, delta_y = optical_flow_pixel_ori(Ix, Iy, It, pt, kernel_size)

    good_new = p0[0]
    good_new[0, 0] = pt[0] + delta_x
    good_new[0, 1] = pt[1] + delta_y

    return good_new

def optical_flow_with_difference(old_gray, new_gray, diff, p0, kernel_size):

    I1_smooth = cv2.GaussianBlur(old_gray  # input image
                                 , (3, 3)  # shape of the kernel
                                 , 0  # lambda
                                 )
    I2_smooth = cv2.GaussianBlur(new_gray, (3, 3), 0)
    Ix = signal.convolve2d(I1_smooth, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, 0.25],
                                                                                                   [-0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in Y direction
    Iy = signal.convolve2d(I1_smooth, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, -0.25],
                                                                                                   [0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in XY direction
    It = signal.convolve2d(I1_smooth, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                [[-0.25, -0.25],
                                                                                                 [-0.25, -0.25]],
                                                                                                'same')
    num = len(p0)
    good_new = []
    for n in range(num):
        pt = [p0[n, 0, 0], p0[n, 0, 1]]
        delta_x, delta_y = optical_flow_pixel_ori(Ix, Iy, It, pt, kernel_size)
        good_new.append([pt[0] + delta_x, pt[1] + delta_y])

    return good_new

def dense_flow_with_difference(old_gray, new_gray, diff, p0, kernel_size):

    I1_smooth = cv2.GaussianBlur(old_gray  # input image
                                 , (3, 3)  # shape of the kernel
                                 , 0  # lambda
                                 )
    I2_smooth = cv2.GaussianBlur(new_gray, (3, 3), 0)
    Ix = signal.convolve2d(I1_smooth, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, 0.25],
                                                                                                   [-0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in Y direction
    Iy = signal.convolve2d(I1_smooth, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, -0.25],
                                                                                                   [0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in XY direction
    It = signal.convolve2d(I1_smooth, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                [[-0.25, -0.25],
                                                                                                 [-0.25, -0.25]],
                                                                                                'same')
    mag = np.zeros_like(new_gray, dtype= np.float32) # magnitude
    ang = np.zeros_like(new_gray, dtype= np.float32) # angle
    num = len(p0)
    good_new = []

    #print(Ix, Iy, It)
    height, width = new_gray.shape
    for i in range(kernel_size//2, height - kernel_size//2):
        for j in range(kernel_size//2, width - kernel_size//2):
            pt = [i, j]
            #print(i, j)
            u, v = optical_flow_pixel_ori(Ix, Iy, It, pt, kernel_size)
            mag[i, j] = np.sqrt(u**2 + v**2)
            ang[i, j] = math.atan2(u, v)
            #print(u[i, j], v[i, j])


    '''
    colors = "bgrcmykw"
    color_index = random.randrange(0, 8)
    c = colors[color_index]
    # ======= Plotting the vectors on the image========
    plt.subplot(1, 1, 1)
    plt.title('Vector plot of Optical Flow of good features')

    step = 1
    plt.quiver(np.arange(0, flow.shape[1], step), np.arange(flow.shape[0], 0, -step),
           flow[::step, ::step, 0], flow[::step, ::step, 1])
    #plt.imshow(It, cmap=cm.gray)
    
    for i in range(height):
        for j in range(width):
            if abs(u[i, j]) > 0 or abs(v[i, j]) > 0:  # setting the threshold to plot the vectors
                plt.arrow(j, i, v[i, j], u[i, j], head_width=5, head_length=5, color=c)
    
    plt.show()
    '''
    return mag, ang

def LK_OpticalFlow(Image1,  # Frame 1
                   Image2,  # Frame 1
                   ):
    '''
    This function implements the LK optical flow estimation algorithm with two frame data and without the pyramidal approach.
    '''
    I1 = np.array(Image1)
    I2 = np.array(Image2)
    S = np.shape(I1)

    # applying Gaussian filter of size 3x3 to eliminate any noise
    I1_smooth = cv2.GaussianBlur(I1  # input image
                                 , (3, 3)  # shape of the kernel
                                 , 0  # lambda
                                 )
    I2_smooth = cv2.GaussianBlur(I2, (3, 3), 0)

    '''
    let the filter in x-direction be Gx = 0.25*[[-1,1],[-1,1]]
    let the filter in y-direction be Gy = 0.25*[[-1,-1],[1,1]]
    let the filter in xy-direction be Gt = 0.25*[[1,1],[1, 1]]
    **1/4 = 0.25** for a 2x2 filter
    '''

    # First Derivative in X direction
    Ix = signal.convolve2d(I1_smooth, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, 0.25],
                                                                                                   [-0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in Y direction
    Iy = signal.convolve2d(I1_smooth, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                  [[-0.25, -0.25],
                                                                                                   [0.25, 0.25]],
                                                                                                  'same')
    # First Derivative in XY direction
    It = signal.convolve2d(I1_smooth, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2_smooth,
                                                                                                [[-0.25, -0.25],
                                                                                                 [-0.25, -0.25]],
                                                                                                'same')

    # finding the good features
    features = cv2.goodFeaturesToTrack(I1_smooth  # Input image
                                       , 10000  # max corners
                                       , 0.01  # lambda 1 (quality)
                                       , 10  # lambda 2 (quality)
                                       )

    feature = np.int0(features)

    plt.subplot(1, 3, 1)
    plt.title('Frame 1')
    plt.imshow(I1_smooth, cmap=cm.gray)
    plt.subplot(1, 3, 2)
    plt.title('Frame 2')
    plt.imshow(I2_smooth, cmap=cm.gray)  # plotting the features in frame1 and plotting over the same
    for i in feature:
        x, y = i.ravel()
        cv2.circle(I1_smooth  # input image
                   , (x, y)  # centre
                   , 3  # radius
                   , 0  # color of the circle
                   , -1  # thickness of the outline
                   )

    # creating the u and v vector
    u = v = np.nan * np.ones(S)

    # Calculating the u and v arrays for the good features obtained n the previous step.
    for l in feature:
        j, i = l.ravel()
        # calculating the derivatives for the neighbouring pixels
        # since we are using  a 3*3 window, we have 9 elements for each derivative.

        IX = (
        [Ix[i - 1, j - 1], Ix[i, j - 1], Ix[i - 1, j - 1], Ix[i - 1, j], Ix[i, j], Ix[i + 1, j], Ix[i - 1, j + 1],
         Ix[i, j + 1], Ix[i + 1, j - 1]])  # The x-component of the gradient vector
        IY = (
        [Iy[i - 1, j - 1], Iy[i, j - 1], Iy[i - 1, j - 1], Iy[i - 1, j], Iy[i, j], Iy[i + 1, j], Iy[i - 1, j + 1],
         Iy[i, j + 1], Iy[i + 1, j - 1]])  # The Y-component of the gradient vector
        IT = (
        [It[i - 1, j - 1], It[i, j - 1], It[i - 1, j - 1], It[i - 1, j], It[i, j], It[i + 1, j], It[i - 1, j + 1],
         It[i, j + 1], It[i + 1, j - 1]])  # The XY-component of the gradient vector

        # Using the minimum least squares solution approach
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK))  # transpose of A
        LK = np.array(np.matrix.transpose(LK))

        A1 = np.dot(LK_T, LK)  # Psedudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2, LK_T)

        (u[i, j], v[i, j]) = np.dot(A3, IT)  # we have the vectors with minimized square error

    # ======= Pick Random color for vector plot========
    colors = "bgrcmykw"
    color_index = random.randrange(0, 8)
    c = colors[color_index]
    # ======= Plotting the vectors on the image========
    plt.subplot(1, 3, 3)
    plt.title('Vector plot of Optical Flow of good features')
    plt.imshow(I1, cmap=cm.gray)
    for i in range(S[0]):
        for j in range(S[1]):
            if abs(u[i, j]) > t or abs(v[i, j]) > t:  # setting the threshold to plot the vectors
                plt.arrow(j, i, v[i, j], u[i, j], head_width=5, head_length=5, color=c)

    plt.show()

class NeedleTracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=20,
                              qualityLevel=0.8,
                              minDistance=7,
                              blockSize=50)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(5, 5),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0 = []

    def init_p0(self, init_point):
        p0 = []
        needle_point = []
        needle_point.append(float(init_point[1]))
        needle_point.append(float(init_point[0]))
        #key_feature = []
        #key_feature.append(needle_point)
        p0.append(needle_point)
        self.p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)

    def track(self, old_gray, new_gray):
        #old_gray = cv2.resize(old_gray, dsize=(0, 0), fx = 0.25, fy = 0.25, interpolation=cv2.INTER_CUBIC)
        #new_gray = cv2.resize(new_gray, dsize=(0, 0), fx = 0.25, fy = 0.25, interpolation=cv2.INTER_CUBIC)
        #good_new = optical_flow(old_gray, new_gray, self.p0, self.lk_params)
        #good_new = optical_flow_ori(old_gray, new_gray, self.p0 / 4, 3)
        #if len(good_new) == 0:
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
        return [new_y, new_x]

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
        feature_params = dict(maxCorners=20,
                              qualityLevel=0.8,
                              minDistance=7,
                              blockSize=50)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=4,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0 = []
        self.point_num = 0

    def init_p0(self, init_points):
        p0 = []
        self.point_num = len(init_points)
        for n in range(self.point_num):
            #key_feature = []
            #key_feature.append(needle_point)
            for p in init_points[n]:
                print(p)
                p0.append([p[1], p[0]])
        self.p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)

    def optical_track(self, old_gray, new_gray):
        #p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, self.p0, None, **self.lk_params)
        #print(st.shape)
        good_new = optical_flow(old_gray, new_gray, self.p0, self.lk_params)
        if len(good_new) == 0:
            good_new = self.p0

        #good_new = int(np.round(good_new))

        self.p0 = good_new.reshape(-1, 1, 2)
        return good_new

    def dense_track(self, old_im, new_im):
        hsv = np.zeros_like(old_im)
        hsv[..., 1] = 255
        prvs = cv2.cvtColor(old_im,cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 1, 15, 1, 5, 1.2, 0)
        #u, v = dense_flow_with_difference(prvs, next, None, self.p0, 15)
        #print(flow[..., 0], u)
        #print(flow[..., 0].shape, u.shape)
        u = flow[..., 0]
        v = flow[..., 1]
        u[u < 0.1] = 0
        v[u < 0.1] = 0
        mag, ang = cv2.cartToPolar(u, v)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(1) & 0xff
        if k == ord('s'):
            cv2.imwrite('./tracking_saved/opticalfb.png', new_im)
            cv2.imwrite('./tracking_saved/opticalhsv.png', rgb)

        return rgb

    def generate_bg(self, previous):
        return np.average(previous, axis = 0)

    def difference_track(self, previous, new_gray, threshold):
        if len(previous)> 5:
            previous = previous[-5::]
        bg = self.generate_bg(previous)
        diff_im = np.zeros_like(new_gray, dtype=uint8)
        difference = np.abs(bg - new_gray)
        threshold = np.sum(np.sum(difference)) / (new_gray.shape[0] + new_gray.shape[1])
        diff_im[difference > threshold] = 255
        diff_median = cv2.medianBlur(diff_im, 3)
        cv2.imshow('difference', diff_median)
        cv2.waitKey(1)

        return difference

    def difference_optical_track(self, previous, new_gray):
        old_gray = previous[-1]
        if len(previous) > 5:
            previous = previous[-5::]
        bg = self.generate_bg(previous)
        diff_im = np.zeros_like(new_gray, dtype=uint8)
        difference = np.abs(bg - new_gray)
        threshold = np.sum(np.sum(difference)) / (new_gray.shape[0] + new_gray.shape[1])
        diff_im[difference < 10] = 0
        old_gray = cv2.resize(old_gray, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
        new_gray = cv2.resize(new_gray, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('difference', old_gray)
        cv2.waitKey(1000)
        good_new = optical_flow_with_difference(old_gray, new_gray, diff_im, self.p0, 3)
        if len(good_new) == 0:
            print("nothing detected")
            good_new = self.p0

        #good_new = int(np.round(good_new))

        self.p0 = np.array(good_new, np.float32).reshape(-1, 1, 2)
        return good_new * 4

    def difference_dense_track(self, previous, new_gray):
        old_gray = previous[-1]
        if len(previous) > 5:
            previous = previous[-5::]
        bg = self.generate_bg(previous)
        diff_im = np.zeros_like(new_gray, dtype=uint8)
        difference = np.abs(bg - new_gray)
        threshold = np.sum(np.sum(difference)) / (new_gray.shape[0] + new_gray.shape[1])
        diff_im[difference < 10] = 0
        #old_gray = cv2.resize(old_gray, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
        #new_gray = cv2.resize(new_gray, dsize=(0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)

        #dense_flow_with_difference(old_gray, new_gray, diff_im, self.p0, 3)

        flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        hsv = np.zeros_like(new_gray)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff

        if k == ord('s'):
            cv2.imwrite('./tracking_saved/opticalfb.png', new_gray)
            cv2.imwrite('./tracking_saved/opticalhsv.png', rgb)

        return self.p0

class ParticleFilter:
    def __init__(self, particle_num):
        self.first_gray = None
        self.boxes0 = []
        self.last_boxes = []
        self.new_boxes = []
        self.blobs0 = []
        self.new_blobs = []
        self.object_num = 0
        self.particles0 = []
        self.new_particles = []
        self.particle_num = particle_num
        self.difference_flags = []

    def init_boxes0(self, first_gray, init_boxes, init_blobs):
        """
        :param init_boxes: center_Y, center_X, height, width
        :return:
        """
        self.first_gray = first_gray
        self.object_num = len(init_boxes)
        self.particles0 = []
        self.new_particles = []
        self.boxes0 = []
        self.blobs0 = init_blobs
        #print(self.object_num, self.particle_num)
        for b, bl in zip(init_boxes, init_blobs):
            self.boxes0.append(b[:2])
            #generated_particles = create_Gaussian_particles(mean = b[:2], std = [5, 5], N = self.particle_num)
            generated_particles = self.create_particles_within_blob(bl, N=self.particle_num)
            #print(generated_particles)
            self.particles0.append(generated_particles)

        self.new_boxes = self.boxes0
        self.last_boxes = self.boxes0
        self.new_blobs = self.blobs0
        self.new_particles = self.particles0
        difference_flags = np.zeros(len(self.new_boxes))
        self.difference_flags = difference_flags > 1


    def create_particles_within_blob(self, blob, N):
        num_pixel = blob.shape[0]
        particles_indexes = np.random.randint(0, num_pixel, N, dtype = np.int32)

        particles = np.empty((N, 3))
        particles[:, :2] = blob[particles_indexes, :]
        particles[:, 2] = np.ones((N), dtype=np.float32) / N
        return particles

    def generate_bg(self, previous):
        return np.average(previous, axis = 0)

    def resampling(self, threshold, std):
        for n in range(self.object_num):
            last_box = self.last_boxes[n]
            if not self.difference_flags[n]:
                continue
            box = self.new_boxes[n]
            move_vector0 = box[0] - last_box[0]
            move_vector1 = box[1] - last_box[1]
            #print("speed", move_vector0, move_vector1)
            left_particles = []
            for n_p in range(self.particle_num):
                if self.new_particles[n][n_p][2] > threshold:
                    left_particles.append(self.new_particles[n][n_p])
                #else:
                    #new_particle_0 = self.new_particles[n][n_p][0] + move_vector0
                    #new_particle_1 = self.new_particles[n][n_p][1] + move_vector1
                    #new_particle = np.zeros((1, 3))
                    #new_particle[0, 0] = new_particle_0
                    #new_particle[0, 1] = new_particle_1
                    #print(new_particle.shape, self.new_particles[n][n_p].shape)
                    #left_particles.extend(new_particle)
            #if len(left_particles) < self.particle_num:
                #new_particles = create_Gaussian_particles(mean=box[:2], std=[5, 5], N=(self.particle_num - len(left_particles)))
                #left_particles.extend(new_particles)
            left_num = len(left_particles)
            #print("left_num", left_num)
            if left_num < self.particle_num:
                new_particles = create_Gaussian_particles(mean=box[:2] + [move_vector0, move_vector1], std=[std, std], N=(self.particle_num - len(left_particles)))
                left_particles.extend(new_particles)
            else:
                for i in range(self.particle_num - left_num):
                    selected_left = np.random.randint(low=0, high=left_num, size=3)

                    new_particle_0 = np.average(np.array(left_particles)[selected_left, 0])
                    new_particle_1 = np.average(np.array(left_particles)[selected_left, 1])
                    new_particle = np.zeros((1, 3))
                    new_particle[0, 0] = new_particle_0
                    new_particle[0, 1] = new_particle_1
                    # print(new_particle.shape, self.new_particles[n][n_p].shape)
                    left_particles.extend(new_particle)
            self.new_particles[n] = left_particles

    def resampling_within_blobs(self, blobs):
        self.new_particles.clear()
        for bl in blobs:
            resampled_particles = self.create_particles_within_blob(bl, N=self.particle_num)
            #print(generated_particles)
            self.new_particles.append(resampled_particles)


    def track(self, previous, new_gray, kernel_size, s_thre):
        old_gray = previous[-1]
        #cv2.imshow('old_gray', old_gray)
        #cv2.imshow('new_gray1', new_gray)
        if len(previous) > 5:
            previous = previous[-5::]
        bg = self.generate_bg(previous)
        diff_im = np.zeros_like(new_gray, dtype=uint8)
        difference = np.abs(bg - new_gray)


        threshold = 10 #np.sum(np.sum(difference)) / (new_gray.shape[0] + new_gray.shape[1])
        diff_im[difference < threshold] = 0
        diff_im[difference > threshold] = 255
        diff_im = cv2.medianBlur(diff_im, 3)
        #ret, th = cv2.threshold(new_gray, 130, 255, cv2.THRESH_BINARY)
        #binary = np.zeros(th.shape, np.uint8)
        #binary[np.where(th == 0)] = 255
        #binary[np.where(th == 255)] = 0
        new_boxes0 = []
        all_left_particles = []

        #print(self.difference_flags)
        flag_ind = 0
        num_diffs = []
        for box, pars0, pars in zip(self.new_boxes, self.particles0, self.new_particles):
            left_particles = []
            if not self.difference_flags[flag_ind]:
                pars_array = np.array(pars, np.int)
                difference_flag = diff_im[pars_array[:, 0], pars_array[:, 1]]

                self.difference_flags[flag_ind] = np.sum(difference_flag) > 0
                #print("difference_flag", flag_ind, self.difference_flags[flag_ind])
            num_diff = 0
            for x0, x in zip(pars0, pars):
                p_h0 = int(x0[0])
                p_w0 = int(x0[1])
                feature_ymin0 = p_h0 - kernel_size // 2
                feature_ymax0 = p_h0 + kernel_size // 2 + 1
                feature_xmin0 = p_w0 - kernel_size // 2
                feature_xmax0 = p_w0 + kernel_size // 2 + 1

                p_h = int(x[0])
                p_w = int(x[1])
                feature_ymin = p_h - kernel_size//2
                feature_ymax = p_h + kernel_size//2 + 1
                feature_xmin = p_w - kernel_size//2
                feature_xmax = p_w + kernel_size//2 + 1
                #cv2.imshow('block', old_gray[feature_ymin:feature_ymax, feature_xmin:feature_xmax])
                #cv2.waitKey(0)
                #similarity = eucli(old_gray[feature_ymin:feature_ymax, feature_xmin:feature_xmax], new_gray[feature_ymin:feature_ymax, feature_xmin:feature_xmax])

                if self.difference_flags[flag_ind] > 0:
                    #prob = similarity#*0.5 + diff_im[p_h, p_w] / 255.0 *0.5#cosine(old_gray[feature_ymin:feature_ymax, feature_xmin:feature_xmax], new_gray[feature_ymin:feature_ymax, feature_xmin:feature_xmax])
                    prob = diff_im[p_h, p_w] / 255.0



                    #prob = 1 - similarity
                else:
                    #prob = similarity
                    prob = diff_im[p_h, p_w] / 255.0

                if prob > 0: # means there is movement
                    num_diff += 1
                #if similarity < 0.5:
                #print(similarity, prob, box)
                #print(p_h, p_w, diff_im[p_h, p_w])
                left_particles.append([p_h, p_w, prob])
            num_diffs.append(num_diff)
            all_left_particles.append(left_particles)

            left_particles = np.array(left_particles)
            all_similarities = left_particles[:, 2].sum()
            if all_similarities:
                #print("with particles useful")
                y_ave = (left_particles[:, 0] * left_particles[:, 2]).sum() / all_similarities
                x_ave = (left_particles[:, 1] * left_particles[:, 2]).sum() / all_similarities
                #print("x_ave", x_ave, all_similarities)
                new_boxes0.append([y_ave, x_ave])
            else:
                new_boxes0.append(box)
            flag_ind += 1
        self.new_particles = all_left_particles
        self.last_boxes = self.new_boxes
        self.new_boxes = new_boxes0
        #print(self.boxes0)
        s_thre = 0.3
        self.resampling(s_thre, 7)
        #new_gray_particles = new_gray.copy()
        #new_gray_particles = draw_particles(new_gray_particles, self.particles)


        return new_boxes0, diff_im, num_diffs#, new_gray_particles


def cosine(m1, m2):
    m1 = m1.flatten()  + Epsilon
    m2 = m2.flatten()  + Epsilon
    print(m1, m2)
    dividor = m1 + m2
    m1 = np.divide(m1, dividor)
    m2 = np.divide(m2, dividor)
    m1_norm = np.linalg.norm(m1)
    m2_norm = np.linalg.norm(m2)
    #eucli_dst = np.linalg.norm(m1 - m2)

    n = m1.shape[0]
    cos = cosine_similarity(m1.reshape(1, -1), m2.reshape(1, -1))
    #scale = 1 - eucli_dst / (m1_norm + m2_norm)
    scale = m1_norm / m2_norm
    if scale > 1:
        scale = 1 / scale

    similarity = cos * 0.5 + scale * 0.5
    #if similarity< 0.5:
    print("----------------", m1, m2, cos, scale, similarity, "----------------")
    return similarity

def eucli(m1, m2):
    m1 = m1.flatten() + Epsilon
    m2 = m2.flatten() + Epsilon
    n = m1.shape[0]
    dividor = m1 + m2
    m1 = np.divide(m1, dividor)
    m2 = np.divide(m2, dividor)

    dst = np.sum(np.abs(m1 - m2)) / n
    dst = np.max(np.abs(m1 - m2))

    #dst = distance.euclidean(m1, m2)
    similarity = 1 - dst #dst / math.sqrt(n * 255.0**2)
    #if (1-similarity) < 0:
    #print("-------------", m1, m2, similarity, "------------------------")
    return similarity

def tanh(m1, m2):
    m1 = m1.flatten() / 255.0
    m2 = m2.flatten() / 255.0

    dst = distance.euclidean(m1, m2)
    similarity = 1 / (1 + np.exp(dst))
    if similarity < 0.5:
        print("-------------", m1, m2, similarity, dst, "------------------------")
    return similarity

def draw_particles(im, particles):
    for ps in particles:
        for p in ps:
            im = cv2.rectangle(im, (int(p[1]), int(p[0])), (int(p[1]), int(p[0])), color = (100,100,255), thickness=1)
    return im

