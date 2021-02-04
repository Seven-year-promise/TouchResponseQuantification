import numpy as np
from easydict import EasyDict as edict
import yaml_1
import math
import cv2

def well_detection(im, gray, threshold = 50):
    # gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 5,
                               param1=240, param2=50,
                               minRadius=95, maxRadius=105)
    #print(circles)
    """
    muted when training
    if circles is not None:
        circles_int = np.uint16(np.around(circles))
        for i in circles_int[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(gray, center, 1, (0, 255, 0), 3)
            # circle outline
            radius = i[2]
            cv2.circle(gray, center, radius, (0, 255, 0), 3)
    """
    #cv2.imshow("detected circles", gray)
    #cv2.waitKey(1000)
    if circles is not None:
        well_centerx = np.uint16(np.round(np.average(circles[0, :, 0])))
        well_centery = np.uint16(np.round(np.average(circles[0, :, 1])))
        well_radius = 115 #np.uint16(np.round(np.average(circles[0, :, 2])))
        #return True, (well_centerx, well_centery, 110)


    else:
        well_centerx = 240
        well_centery = 240
        well_radius = 115
        #return False, (240, 240, 110)

    # first rough mask for well detection
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    # second fine-tuned mask
    ret, th = cv2.threshold(gray_masked, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((10, 10), dtype=np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    im_closing = cv2.bitwise_and(im, im, mask=closing)

    white_indexes = np.where(closing == 255)
    well_centery = int(np.round(np.average(white_indexes[0])))
    well_centerx = int(np.round(np.average(white_indexes[1])))
    # third fine-tuned mask for background white
    closing_inv = cv2.bitwise_not(closing)
    closing_inv = np.array((closing_inv, closing_inv, closing_inv)).transpose(1, 2, 0)
    im_closing_inv = closing_inv + im_closing

    #cv2.circle(gray, (well_centerx, well_centery), 1, (0, 255, 0), 5)
    #cv2.imshow("detected circles", im_closing_inv)
    #cv2.waitKey(1000)

    return True, (well_centerx, well_centery, well_radius), im_closing_inv

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def kpt_affine(kpt, mat):
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)

def Config(filename):

    with open(filename, 'r') as f:
        parser = edict(yaml_1.load(f))
    for x in parser:
        print( '{}: {}'.format(x, parser[x]))
    return parser

def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=None):

    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    if multiple != None:
	    for i, param_group in enumerate(optimizer.param_groups):
		    param_group['lr'] = lr * multiple[i]
    else:
	    for i, param_group in enumerate(optimizer.param_groups):
		    param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
