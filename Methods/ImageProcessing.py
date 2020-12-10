import cv2
import numpy as np
from Methods.FeatureExtraction import SIFT, Binarization



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
    kernel = np.ones((50, 50), dtype=np.uint8)
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



def feature_extraction(ori_im, threshold = None, method = "Otsu", well_infos = None):
    #.sift.detect_keypoints(ori_im)
    #im_with_keypoints = .sift.drawpoints(ori_im)
    if method == "sift":
        sift = SIFT()
        needle, _ = sift.compute_sift(ori_im)
    else:
        binarize = Binarization()
        binarize.method = method
        if method == "Binary":
            needle, fish = binarize.compute_binary(ori_im, well_infos)
        elif method == "Otsu":
            needle, fish = binarize.compute_binary(ori_im, well_infos)
        elif method == "RG":
            needle, fish = binarize.compute_binary(ori_im, threshold)
        elif method == "LRB":
            needle, fish = binarize.compute_binary(ori_im, well_infos)
        else:
            print("please select one method for feature extraction")
            needle, fish = ori_im, ori_im
    return needle, fish, None #im_with_keypoints, .sift.keypoints, .sift.descriptors

def meanshift_seg( ori_im):
    # TO DO
    pass

"""
def blob_detection(, ori, binary, size_threshold_High = 100, size_threshold_Low = 10):
    ret, markers = cv2.connectedComponents(binary)
    number_objects = np.max(markers)
    print(number_objects)
    centers = []
    final_blobs = []
    for n in range(1, number_objects + 1):
        blob = np.where(markers == n)
        blobarray = np.ones((len(blob[0]), 2), dtype=np.int)
        blobarray[:, 0] = blob[0]
        blobarray[:, 1] = blob[1]
        width = np.max(blobarray[:, 1]) - np.min(blobarray[:, 1])
        height = np.max(blobarray[:, 0]) - np.min(blobarray[:, 0])
        if width > size_threshold_Low and width < size_threshold_High:
            if height > size_threshold_Low and height < size_threshold_High:
                final_blobs.append(blobarray)
                cy = (int)(np.round(np.average(blob[0])))
                cx = (int)(np.round(np.average(blob[1])))
                centers.append([cy, cx])

    return ori, final_blobs, centers

"""

def region_grow( ori_im, points, threshold):
    h, w = ori_im.shape
    binary = np.zeros((h, w), np.uint8)
    im_copy = ori_im.copy()
    for point in points:
        print(point, points)
        y0, x0 = point
        i, j = 1, 1 # i:y, j:x
        binary_pts = []
        binary_pt = ori_im[y0, x0] < threshold
        if binary_pt:
            binary_pts.append(binary_pt)
            binary[y0, x0] = 255
        print(len(binary_pts))
        while len(binary_pts) > 0:
            pt_num = 2*(2*i+1) + 2*(2*j+1)
            edge_points = np.zeros((pt_num, 2), np.int)

            xmin = x0 - j
            xmax = x0 + j
            ymin = y0 - i
            ymax = y0 + i
            edge_points[:(2 * j + 1), 0] = np.repeat(ymin, (2 * j + 1))
            edge_points[:(2 * j + 1), 1] = np.arange(xmin, xmax + 1)
            edge_points[(2 * j + 1):(2 * j + 2 * i + 2), 0] = np.arange(ymin, ymax + 1)
            edge_points[(2 * j + 1):(2 * j + 2 * i + 2), 1] = np.repeat(xmax, (2 * i + 1))
            edge_points[(2 * j + 2 * i + 2):(2 * j + 4 * i + 3), 0] = np.arange(ymin, ymax + 1)
            edge_points[(2 * j + 2 * i + 2):(2 * j + 4 * i + 3), 1] = np.repeat(xmin, (2 * i + 1))
            edge_points[(2 * j + 4 * i + 3):(4 * j + 4 * i + 4), 0] = np.repeat(ymax, (2 * j + 1))
            edge_points[(2 * j + 4 * i + 3):(4 * j + 4 * i + 4), 1] = np.arange(xmin, xmax + 1)
            binary_pts.clear()
            for y, x in zip(edge_points[:, 0], edge_points[:, 1]):
                binary_pt = ori_im[y, x] < threshold
                display = cv2.rectangle(im_copy, (x, y), (x, y), color=[0, 255, 0], thickness=2)
                cv2.imshow("region", display)
                cv2.waitKey(50)
                if binary_pt:
                    binary_pts.append(binary_pt)
                    binary[y, x] = 255
            print(len(binary_pts))
            i += 1
            j += 1
    return binary

def region_growing(img, seed):
    # Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_threshold = 0.2
    region_size = 1
    intensity_difference = 0
    neighbor_points_list = []
    neighbor_intensity_list = []

    # Mean of the segmented region
    region_mean = img[seed]

    # Input image parameters
    height, width = img.shape
    image_size = height * width

    # Initialize segmented output image
    segmented_img = np.zeros((height, width, 1), np.uint8)

    # Region growing until intensity difference becomes greater than certain threshold
    while (intensity_difference < region_threshold) & (region_size < image_size):
        # Loop through neighbor pixels
        for i in range(4):
            # Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            # Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)

            # Add neighbor if inside and not already in segmented_img
            if check_inside:
                if segmented_img[x_new, y_new] == 0:
                    neighbor_points_list.append([x_new, y_new])
                    neighbor_intensity_list.append(img[x_new, y_new])
                    segmented_img[x_new, y_new] = 255

        # Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list - region_mean)
        pixel_distance = min(distance)
        index = np.where(distance == pixel_distance)[0][0]
        segmented_img[seed[0], seed[1]] = 255
        region_size += 1

        # New region mean
        region_mean = (region_mean * region_size + neighbor_intensity_list[index]) / (region_size + 1)

        # Update the seed value
        seed = neighbor_points_list[index]
        # Remove the value from the neighborhood lists
        neighbor_intensity_list[index] = neighbor_intensity_list[-1]
        neighbor_points_list[index] = neighbor_points_list[-1]

    return segmented_img


def blob_detection(ori_im, binary, local_threshold = 190):
    #image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    binary = cv2.bitwise_not(binary)
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    init_points = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            #ori_im = cv2.drawContours(ori_im, contour, -1, (0, 255, 0), 1)
            contour = np.array(contour)
            xmin = np.min(contour[:, 0, 0])
            xmax = np.max(contour[:, 0, 0])
            ymin = np.min(contour[:, 0, 1])
            ymax = np.max(contour[:, 0, 1])

            block = ori_im[ymin:ymax, xmin:xmax]
            #indexes = np.where(ori_im[ymin:ymax, xmin:xmax] < 100)
            threshold = np.min(np.array(block, dtype=np.int))
            min_index = np.where(block == threshold)
            #print(min_index)
            #cv2.imshow("binary", ori_im[ymin:ymax, xmin:xmax])
            #cv2.waitKey(10000)
            #print(indexes, indexes[1][0], indexes[0][0])
            #ori_im = cv2.rectangle(ori_im, (xmin+indexes[1][0], ymin+indexes[0][0]), (xmin+indexes[1][0], ymin+indexes[0][0]), color=(0, 255, 0), thickness=5)
            #cv2.imshow("binary", ori_im)
            #cv2.waitKey(10000)
            init_points.append([ymin+min_index[0][0], xmin+min_index[1][0]])
    new_binary = region_grow(ori_im, init_points, local_threshold)
    cv2.imshow("region based binary", new_binary)
    cv2.waitKey(10000)
    #ori_im = cv2.rectangle(ori_im, (xmin, ymin), (xmax, ymax), color=[0, 255, 0], thickness=3)

    return ori_im, None, None


