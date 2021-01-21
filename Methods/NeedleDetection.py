import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.feature import hog
from skimage.morphology import skeletonize
from xml_reader import XML_Reader

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

def blobDetection(binary,
                  dis_threshold,
                  size_threshold_High = 200,
                  size_threshold_Low = 10):
    height = binary.shape[0]
    width = binary.shape[1]
    blobs = []
    for h in range(height):
        for w in range(width):
            if binary[h, w] > 0:
                if len(blobs)>0:
                    flag = False
                    for bi in range(len(blobs)):
                        for position in blobs[bi]:
                            ch = position[0]
                            cw = position[1]
                            distance = (h-ch)*(h-ch) + (w-cw)*(w-cw)
                            if distance < dis_threshold*dis_threshold:
                                blobs[bi].append([h, w])
                                flag=True
                                break
                        if flag:
                            break
                    if not flag:
                        new_blob = [[h, w]]
                        blobs.append(new_blob)
                else:
                    new_blob = [[h, w]]
                    blobs.append(new_blob)
    centers = []
    final_blobs = []
    for b in blobs:
        if (len(b)>size_threshold_Low) & (len(b)<size_threshold_High):
            final_blobs.append(b)
            center = np.round(np.average(np.array(b), axis=0))

            centers.append(center)
    return  final_blobs, centers

def blob_counter(masked, dis_threshold, size_threshold_High, size_threshold_Low, what_detected = 'needle'):
    ret, markers = cv2.connectedComponents(masked)
    number_objects = np.max(markers)
    centers = []
    final_blobs = []
    for n in range(1, number_objects+1):
        blob = np.where(markers==n)
        blobarray = np.ones((len(blob[0]), 2), dtype = np.int)
        blobarray[:, 0] = blob[0]
        blobarray[:, 1] = blob[1]
        width = np.max(blobarray[:, 1]) - np.min(blobarray[:, 1])
        height = np.max(blobarray[:, 0]) - np.min(blobarray[:, 0])
        if width>size_threshold_Low and width<size_threshold_High:
            if height>size_threshold_Low and height<size_threshold_High:
                final_blobs.append(blobarray)
                cy = (int)(np.round(np.average(blob[0])))
                cx = (int)(np.round(np.average(blob[1])))
                centers.append([cy, cx])

    return  final_blobs, centers

def maxima_blob_counter(binary):

    ret, markers = cv2.connectedComponents(binary)
    number_objects = np.max(markers)
    num_pixels = []
    num_pixels.append(0)
    for n in range(1, number_objects+1):
        blob = np.where(markers==n)
        num_pixels.append(len(blob[0]))
    maxima_index = np.argmax(num_pixels)
    binary[np.where(markers!=maxima_index)] = 0
    binary[np.where(markers==maxima_index)] = 1
    return  binary

def object_detection(gray, well_info,
                           threshold = 30,
                           dis_threshold = 10,
                           size_threshold_High = 200,
                           size_threshold_Low = 10,
                           what_detected = 'needle',
                           blur = False): # default for needle detection
    if blur:
        gray = cv2.medianBlur(gray, 5)
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_info[0], well_info[1]), well_info[2], 255, -1)

    masked_display = cv2.bitwise_and(gray, gray, mask=mask)
    #cv2.imshow("masked_display", masked_display)

    candidate = [0, 0]
    if what_detected == 'needle':
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        gray_masked[np.where(gray_masked==0)] = 255
        #cv2.imshow("binary", gray_masked)
        #cv2.waitKey(0)
        min_index = 0

        threshold = np.min(np.array(gray_masked, dtype=np.int))

        min_index = np.where(gray_masked==threshold)
        cy = (int)(np.round(np.average(min_index[0])))
        cx = (int)(np.round(np.average(min_index[1])))
        candidate = [[cy, cx]]
        threshold = threshold+10

    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    reversed_binary = np.bitwise_not(binary)



    masked = cv2.bitwise_and(reversed_binary, reversed_binary, mask=mask)
    #dilation = cv2.dilate(img,kernel,iterations = 1)
    kernel = np.ones((5,5), dtype = np.uint8)
    closing = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow("binary", closing)
    # blobs, centers = blobDetection(masked, dis_threshold, size_threshold_High, size_threshold_Low)
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                                         sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(masked, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('masked', fontsize=20)

    ax[1].imshow(closing, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('closing', fontsize=20)

    #plt.show()
    """
    blobs, centers = blob_counter(closing, dis_threshold, size_threshold_High, size_threshold_Low)

    if what_detected == 'needle':
        blob_num = len(centers)
        if blob_num > 1:
            #blobs = None
            #centers = needle_GaussianOptimization(gray, centers, size = 11)
            blob_cnts = []
            for i in range(blob_num):
                blob_cnts.append(len(blobs[i]))
            max_inx = np.argmax(blob_cnts)
            blobs = [blobs[max_inx]]
            centers = [centers[max_inx]]
        if blob_num == 0:
            blobs = None
            centers = candidate

    return blobs, centers

def fish_detection(gray, well_info,
                   needle_center,
                   threshold = 30,
                   dis_threshold = 10,
                   size_threshold_High = 200,
                   size_threshold_Low = 10,
                   what_detected = 'fish',
                   blur = False): # default for needle detection
    if blur:
        gray = cv2.medianBlur(gray, 5)
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_info[0], well_info[1]), well_info[2], 255, -1)
    candidate = [0, 0]

    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    reversed_binary = np.bitwise_not(binary)



    masked = cv2.bitwise_and(reversed_binary, reversed_binary, mask=mask)
    #dilation = cv2.dilate(img,kernel,iterations = 1)
    kernel = np.ones((5,5), dtype = np.uint8)
    closing = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel)
    median = cv2.medianBlur(closing, 3)
    kernel = np.ones((3,3), dtype = np.uint8)
    dilation = cv2.dilate(median, kernel, iterations = 1)

    #cv2.imshow("binary", masked)
    #cv2.waitKey(0)

    #cv2.imshow("fish_binary", dilation)
    """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 4),
                                     sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(masked, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('masked', fontsize=20)

    ax[1].imshow(closing, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('closing', fontsize=20)

    ax[2].imshow(median, cmap=plt.cm.gray)
    ax[2].axis('off')
    ax[2].set_title('median', fontsize=20)

    ax[3].imshow(dilation, cmap=plt.cm.gray)
    ax[3].axis('off')
    ax[3].set_title('dilation', fontsize=20)

    #plt.show()
    """
    # blobs, centers = blobDetection(masked, dis_threshold, size_threshold_High, size_threshold_Low)
    blobs, centers = blob_counter(closing, dis_threshold, size_threshold_High, size_threshold_Low)
    number_result = len(centers)
    if number_result == 0:
        return None, None
    elif number_result == 1:
        return blobs, centers
    else:
        final_blobs = []
        final_centers = []
        blob_cnt = []
        distances = []
        for i in range(number_result):
            Fcenter = centers[i]
            dis = (needle_center[0] - Fcenter[0]) * (needle_center[0] - Fcenter[0]) + (needle_center[1] - Fcenter[1])*(needle_center[1] - Fcenter[1])
            distances.append(dis)
            if dis > 10*10:
                final_blobs.append(blobs[i])
                final_centers.append(Fcenter)
                blob_cnt.append(len(blobs[i]))
        #cv2.waitKey(0)
        #print(dis, blob_cnt)
        if len(final_centers) == 0:
             max_index = np.argmax(distances)
             return [blobs[max_index]], [centers[max_index]]
        else:
            max_index = np.argmax(blob_cnt)
            return [final_blobs[max_index]], [final_centers[max_index]]

def gaussian_kernel(size=6, sigma=1.0, mu=0.0):
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def needle_GaussianOptimization(gray, candidates, size = 7):
    """
    given an image, find the lowest pixel
    return: h, w
    """
    size
    g_kernel = gaussian_kernel(size = size)
    g_results = []
    for c in candidates:
        offset = (int)(size/2)
        im_batch = gray[(c[0]-offset):(c[0]+offset+1), (c[1]-offset):(c[1]+offset+1)]
        """
        orb = cv2.ORB()

        # find the keypoints with ORB
        kp = orb.detect(im_batch,None)
        print(kp)

        # compute the descriptors with ORB
        kp, des = orb.compute(im_batch, kp)
        print(kp, des)

        corners = cv2.goodFeaturesToTrack(im_batch,25,0.01,10)
        corners = np.int0(corners)
        print(corners)
        """
        #cv2.imshow("im_patch", im_batch)
        dX = cv2.Sobel(im_batch, cv2.CV_32F, 1, 0, (3,3))
        dY = cv2.Sobel(im_batch, cv2.CV_32F, 0, 1, (3,3))
        mag, direction = cv2.cartToPolar(dX, dY, angleInDegrees=True)
        mag_var = np.var(mag.reshape(-1, 1))
        direction_var = np.var(direction.reshape(-1, 1))
        #cv2.imshow("direction", direction)
        #mag_hist = cv2.calcHist([mag],[0],None,[8],[0,256])
        #direction_hist = cv2.calcHist([direction],[0],None,[8],[0,361])
        #plt.hist(direction.ravel(),8,[0,360])
        #plt.show()
        #print("gradient:", mag_var, direction_var)
        #print("gradient finished")
        #fd, hog_image = hog(im_batch, orientations=8, pixels_per_cell=(size, size),
                            #cells_per_block=(1, 1), visualize=True, multichannel=False)
        #cv2.imshow('hog', im_batch)
        #cv2.waitKey(0)
        #print(fd)
        #g_result =  im_batch * g_kernel
        #g_result = np.sum(g_result, axis = 0)
        #g_result = np.sum(g_result, axis = 0)

        g_result = direction_var

        g_results.append(g_result)

    min_index = np.argmin(g_results)
    return [candidates[min_index]]

def needle_usingMaxima(gray, blur = 'false'):
    """
    given an image, find the lowest pixel
    return: h, w
    """
    if blur:
        gray = cv2.medianBlur(gray, 3)
    threshold = np.min(np.array(gray, dtype=np.int))
    min_index = np.where(gray==threshold)
    cy = (int)(np.round(np.average(min_index[0])))
    cx = (int)(np.round(np.average(min_index[1])))
    return (cy, cx)

def detection_using_difference(gray, well_info,
                               diff_order = 2,
                               blur = False): # default for needle detection
    if blur:
        gray = cv2.medianBlur(gray, 5)
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_info[0], well_info[1]), well_info[2], 255, -1)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    height = gray_masked.shape[0]
    width = gray_masked.shape[1]
    difference = np.zeros(gray.shape[:2], dtype="uint8")
    for h in range(height-diff_order):
        for w in range(width-diff_order):
            difference[h, w] = (int)(np.round((np.abs(gray_masked[h, w] - gray_masked[h+2, w+2])/2)))
    cv2.imshow('difference', difference)
    cv2.waitKey(0)

def fish_keypoints(img, well_info, fish_blob, bin_number, gravity_center, scale = 3.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,210)
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_info[0], well_info[1]), well_info[2], 255, -1)
    masked = cv2.bitwise_and(edges, edges, mask=mask)



    blob = np.array(fish_blob)
    cor_min0 = np.min(blob[:, 0])
    cor_min1 = np.min(blob[:, 1])
    cor_max0 = np.max(blob[:, 0])
    cor_max1 = np.max(blob[:, 1])
    #print(cor_min0, cor_min1, cor_max0, cor_max1)
    #print(blob[:, 0], blob[:, 1])

    # ================ find the fish area ================
    scaled_fish_height = (int)((cor_max0 - cor_min0) * scale)
    scaled_fish_width = (int)((cor_max1 - cor_min1) * scale)
    scaled_ymin = (int)((cor_min0+cor_max0-scaled_fish_height)/2.0)
    scaled_ymax = (int)((cor_min0+cor_max0+scaled_fish_height)/2.0)
    scaled_xmin = (int)((cor_min1+cor_max1-scaled_fish_width)/2.0)
    scaled_xmax = (int)((cor_min1+cor_max1+scaled_fish_width)/2.0)

    # =============== process the fish area ===============
    fish_display = img[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax]
    #cv2.imshow("fish_part", fish_display)
    fish = masked[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax]
    #cv2.imshow("fish_edge", fish)

    max_blob = maxima_blob_counter(fish)
    max_blob[max_blob==1] = 255
    kernel = np.ones((5,5), dtype = np.uint8)
    closing = cv2.morphologyEx(max_blob, cv2.MORPH_CLOSE, kernel)
    median = closing #cv2.medianBlur(fish, 5)

    new_fish_blob = np.where(median>0)
    #print(edges)
    #ret, binary = cv2.threshold(fish, 170, 255, cv2.THRESH_BINARY)
    #reversed_binary = np.bitwise_not(binary)

    # ================ new coordinates for gravity and bbox center===================
    (Gx, Gy) = (gravity_center[1], gravity_center[0])#(scaled_xmin + np.average(fish_blobs[1]), scaled_ymin + np.average(fish_blobs[0]))
    #fish_blobs = np.array(fish_blobs)
    new_cor_min0 = scaled_ymin + np.min(new_fish_blob[0])
    new_cor_min1 = scaled_xmin + np.min(new_fish_blob[1])
    new_cor_max0 = scaled_ymin + np.max(new_fish_blob[0])
    new_cor_max1 = scaled_xmin + np.max(new_fish_blob[1])
    #print(scaled_ymin, scaled_xmin, scaled_ymin, scaled_xmin, np.min(new_fish_blob[0]), np.min(new_fish_blob[1]), np.max(new_fish_blob[0]), np.max(new_fish_blob[1]))
    cor_min = (new_cor_min0, new_cor_min1)
    cor_max = (new_cor_max0, new_cor_max1)

    (Cx, Cy) = (np.round((cor_max[1] + cor_min[1])/2.0), np.round((cor_max[0] + cor_min[0])/2.0))

    #print(cor_min, cor_max)
    # =============== skeleton for top_most and bottom_left ===================
    kernel = np.ones((5,5), dtype = np.uint8)
    dilation = cv2.dilate(median, kernel, iterations = 1)

    dilation = maxima_blob_counter(dilation)
    #cv2.imshow("fish_dilation", dilation*255)
    skeleton = skeletonize(dilation)
    skeleton_display = np.array(skeleton*255, np.uint8)
    #cv2.imshow("skeleton", skeleton_display)
    skeleton_cor = np.where(skeleton>0)
    (skeleton_y, skeleton_x) = (skeleton_cor[0], skeleton_cor[1])
    skeleton_minx = np.min(skeleton_x)
    skeleton_miny = np.min(skeleton_y)
    skeleton_maxx = np.max(skeleton_x)
    skeleton_maxy = np.max(skeleton_y)

    cor_min0 = scaled_ymin + skeleton_miny
    cor_min1 = scaled_xmin + skeleton_minx
    cor_max0 = scaled_ymin + skeleton_maxy
    cor_max1 = scaled_xmin + skeleton_maxx

    cor_min = (cor_min0, cor_min1)
    cor_max = (cor_max0, cor_max1)
    skeleton_cor = np.array(skeleton_cor).T
    skeleton_cor[:, 0] += scaled_ymin
    skeleton_cor[:, 1] += scaled_xmin
    #skeleton_display[skeleton_display==0] = 1
    #skeleton_display[skeleton_display==255] = 0
    fish_display[skeleton_display==255] = [0, 255, 0]
    #fish_display[skeleton_display==255, 1] = 255
    #fish_display[skeleton_display==255, 2] = 0
    #cv2.imshow("fish_display_skeleton",  fish_display)
    #print((Gx, Gy), (Cx, Cy), cor_min, cor_max)
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 2),
                                     sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(fish_display, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('fish_display', fontsize=20)
    ax[0].imshow(skeleton, cmap=plt.cm.gray)


    ax[1].imshow(max_blob, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('closing', fontsize=20)

    ax[2].imshow(closing, cmap=plt.cm.gray)
    ax[2].axis('off')
    ax[2].set_title('median', fontsize=20)

    ax[3].imshow(median, cmap=plt.cm.gray)
    ax[3].axis('off')
    ax[3].set_title('dilation', fontsize=20)

    plt.show()
    """

    touching_points = [] # 0 head, 1 body, 2 tail
    if Gx < Cx:
        if Gy < Cy:
            Headx = (int)(np.round(0.75*cor_min[1]+0.25*Gx))
            Heady = (int)(np.round(0.75*cor_min[0]+0.25*Gy))
            Bodyx = (int)(np.round(0.25*cor_min[1]+0.75*Gx))
            Bodyy = (int)(np.round(0.25*cor_min[0]+0.75*Gy))
            Tailx = (int)(np.round(0.75*cor_max[1]+0.25*Gx))
            Taily = (int)(np.round(0.75*cor_max[0]+0.25*Gy))
            touching_points.append([Headx, Heady])
            touching_points.append([Bodyx, Bodyy])
            touching_points.append([Tailx, Taily])
        else:
            Headx = (int)(np.round(0.75*cor_min[1]+0.25*Gx))
            Heady = (int)(np.round(0.75*cor_max[0]+0.25*Gy))
            Bodyx = (int)(np.round(0.25*cor_min[1]+0.75*Gx))
            Bodyy = (int)(np.round(0.25*cor_max[0]+0.75*Gy))
            Tailx = (int)(np.round(0.75*cor_max[1]+0.25*Gx))
            Taily = (int)(np.round(0.75*cor_min[0]+0.25*Gy))
            touching_points.append([Headx, Heady])
            touching_points.append([Bodyx, Bodyy])
            touching_points.append([Tailx, Taily])
    else:
        if Gy < Cy:
            Headx = (int)(np.round(0.75*cor_max[1]+0.25*Gx))
            Heady = (int)(np.round(0.75*cor_min[0]+0.25*Gy))
            Bodyx = (int)(np.round(0.25*cor_max[1]+0.75*Gx))
            Bodyy = (int)(np.round(0.25*cor_min[0]+0.75*Gy))
            Tailx = (int)(np.round(0.75*cor_min[1]+0.25*Gx))
            Taily = (int)(np.round(0.75*cor_max[0]+0.25*Gy))
            touching_points.append([Headx, Heady])
            touching_points.append([Bodyx, Bodyy])
            touching_points.append([Tailx, Taily])
        else:
            Headx = (int)(np.round(0.75*cor_max[1]+0.25*Gx))
            Heady = (int)(np.round(0.75*cor_max[0]+0.25*Gy))
            Bodyx = (int)(np.round(0.25*cor_max[1]+0.75*Gx))
            Bodyy = (int)(np.round(0.25*cor_max[0]+0.75*Gy))
            Tailx = (int)(np.round(0.75*cor_min[1]+0.25*Gx))
            Taily = (int)(np.round(0.75*cor_min[0]+0.25*Gy))
            touching_points.append([Headx, Heady])
            touching_points.append([Bodyx, Bodyy])
            touching_points.append([Tailx, Taily])
    #print(Gx, Gy, Cx, Cy, touching_points)
    keypoints = []
    step_h = np.round((cor_max[0] - cor_min[0])/(bin_number+1))
    step_w = np.round((cor_max[1] - cor_min[1]) / (bin_number + 1))
    for n in range(1, bin_number+1):
        keypoints.append([cor_min[0]+n*step_h, cor_min[1]+n*step_w])

    return keypoints, touching_points, ([cor_min, cor_max]), skeleton_cor, len(new_fish_blob[0])

"""
find the area of the well and focus the area within the well

"""
def circle_detection(gray):
    #gray = cv2.medianBlur(gray, 5)
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
    #cv2.imshow("detected circles", gray)
    #cv2.waitKey(0)
    if circles is not None:
        well_centerx = np.uint16(np.round(np.average(circles[0, :, 0])))
        well_centery = np.uint16(np.round(np.average(circles[0, :, 1])))
        well_radius = np.uint16(np.round(np.average(circles[0, :, 2])*0.9))
        return True, (well_centerx, well_centery, well_radius)
    else:
        return False, (240, 240, 70)

def ComputeIOU(boxA, boxB):
    """
    xmin, ymin, xmax, ymax
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def ComputeOR(boxA, boxB):
    """
    xmin, ymin, xmax, ymax
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea)
    # return the intersection over union value
    return iou

def plot_IOU(IOU_list, fish_num, pt_num = 100):
    thresholds = np.arange(pt_num) / 100.0
    recall_rates = []
    for t in thresholds:
        fish_recall_num = 0
        for iou in IOU_list:
            if iou>t:
                fish_recall_num += 1
        recall_rates.append(fish_recall_num / fish_num)

    plt.plot(thresholds, recall_rates)
    for a, b in zip(thresholds[30:60:20], recall_rates[30:60:20]):
        plt.text(a, b, '(' + str(a) + ', ' + str(round(b, 2)) +')', fontsize=10)
    plt.ylabel("Recall Ratio", fontsize=12)
    plt.xlabel("OR Threshold", fontsize=12)
    #plt.title('Recall Ratio of Larva Detection \nAs Threshold of IoU Varies')
    plt.show()


def ComputeDistance(boxA, boxB):
    """
    xmin, ymin, xmax, ymax
    """
    centerAx = (boxA[0] + boxB[2]) / 2.0
    centerAy = (boxA[1] + boxB[3]) / 2.0
    centerBx = (boxB[0] + boxB[2]) / 2.0
    centerBy = (boxB[1] + boxB[3]) / 2.0
    return np.sqrt((centerAx - centerBx)**2 + (centerAy - centerBy)**2)

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
    for ipath in im_files:
        video_cnt+=1
        # vpath = 'WT_150931_Speed25.avi'#video_files[10]''
        if ipath[-3:] != 'jpg':
            continue
        #if video_cnt >20:
            #break
        print(path + ipath)
        xml_file = xml_base_path + ipath[:-3]+'xml'
        xml_reader.file_path = xml_file
        xml_reader.load_file()
        xml_reader.list_objects()
        ground_truth_needles = xml_reader.needles
        #print(ground_truth_needles)
        ground_truth_fishes = xml_reader.fishes

        im = cv2.imread(path + ipath)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        Well_success, well_info = circle_detection(gray.copy())  # centerx, centery, radius

        #print("\t the well is within:", well_info, Well_success)

        # cv2.imshow("first_frame", frame)
        # cv2.waitKey(0)
        needle_blobs, needle_centers = object_detection(gray, well_info,
                                                        threshold=30,
                                                        dis_threshold=3,
                                                        size_threshold_High=10,
                                                        size_threshold_Low=0,
                                                        what_detected='needle',
                                                        blur=False)

        fish_blobs, fish_centers = fish_detection(gray, well_info,
                                                  needle_center=needle_centers[0],
                                                  threshold=160,
                                                  dis_threshold=4,
                                                  size_threshold_High=100,
                                                  size_threshold_Low=4,
                                                  what_detected='fish',
                                                  blur=False)

        # print("\t the needle center is:", needle_centers)
        # print("\t the fish center is:", fish_centers)
        display_frame = im.copy()
        keypoints = []
        """
        keypoints, touching_points: w, h
        centers, blobs: h,w
        """

        Ncenter = needle_centers[0]
        #display_frame = cv2.circle(display_frame, ((int)(Ncenter[1]), (int)(Ncenter[0])), 2, (0, 0, 255), thickness=2)
        #print("\t the needle center is:", Ncenter, ground_truth_needles)

        # assume that there is only one needle
        if len(ground_truth_needles) != 0:
            gt_needleX = (ground_truth_needles[0][0] + ground_truth_needles[0][2]) / 2.0
            gt_needleY = (ground_truth_needles[0][1] + ground_truth_needles[0][3]) / 2.0

            distance = np.sqrt((Ncenter[0] - gt_needleY)**2 + (Ncenter[1] - gt_needleX)**2)
            print(distance)
            needle_distances += distance
            needle_num += 1
        f_i = 0
        if len(ground_truth_fishes) != 0:
            fish_num += 1
            for Fcenter in fish_centers:
                dis = (Ncenter[0] - Fcenter[0]) * (Ncenter[0] - Fcenter[0]) + (Ncenter[1] - Fcenter[1]) * (
                            Ncenter[1] - Fcenter[1])
                # print("\t the fish candidate center is:", Fcenter)
                if dis > 10 * 10:
                    #print("\t fish center retain", Fcenter)
                    display_frame = cv2.circle(display_frame, ((int)(Fcenter[1]), (int)(Fcenter[0])), 2, (0, 255, 0),
                                               thickness=2)
                    keypoints, touching_points, box, skeleton_cor, num_fish_pixels = fish_keypoints(im, well_info,
                                                                                                    fish_blobs[f_i], 5,
                                                                                                    Fcenter)
                    # display_frame = cv2.rectangle(display_frame, (box[0][1], box[0][0]), (box[1][1], box[1][0]), (0, 255, 255), thickness=1)
                    IOUs = []
                    distances = []
                    for gt_fish in ground_truth_fishes:
                        IOUs.append(ComputeOR(gt_fish, [box[0][1], box[0][0], box[1][1], box[1][0]]))
                        distances.append(ComputeDistance(gt_fish, [box[0][1], box[0][0], box[1][1], box[1][0]]))
                    this_IOU = np.max(IOUs)
                    this_distance = np.min(distances)
                    fish_IOUs += this_IOU
                    fish_distances += this_distance
                    fish_IOU_list.append(this_IOU)
                    print(this_IOU, this_distance)
                    fish_recall_num += 1

                f_i = f_i + 1
        # continue
        # cv2.imshow("first_frame", display_frame)
        # cv2.imwrite("detection_result/" + str(frame_index) +'.png', display_frame)
        # cv2.waitKey(0)
        # print(keypoints)
    print("needle average distance", needle_distances/needle_num)
    print("fish average IOU", fish_IOUs/fish_recall_num)
    print('fish cneter distance', fish_distances/fish_recall_num)
    plot_IOU(fish_IOU_list, fish_num)
    #cv2.imshow("first_frame", display_frame)
    #cv2.waitKey(0)