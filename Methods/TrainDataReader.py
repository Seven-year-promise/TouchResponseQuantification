import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    boxA is the gt_box
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

    plt.figure(figsize=(6, 6))
    plt.plot(thresholds, recall_rates)
    for a, b in zip(thresholds[30:90:20], recall_rates[30:90:20]):
        plt.text(a, b, '(' + str(a) + ', ' + str(round(b, 2)) +')', fontname ="Times New Roman", fontsize = 18)
    plt.ylabel("Ratio of Recall ($R_r$)", fontname ="Times New Roman", fontsize = 18)
    plt.xlabel("The Threshold ($T_o$) of Ratio of Overlap ($R_o$)", fontname ="Times New Roman", fontsize = 18)
    plt.xticks(fontname="Times New Roman", fontsize=18)
    plt.yticks(fontname="Times New Roman", fontsize=18)
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

def ComputeInOut(boxA, boxB):
    """
    boxA: gt_box
        xmin, ymin, xmax, ymax
    """
    centerBx = (boxB[0] + boxB[2]) / 2.0
    centerBy = (boxB[1] + boxB[3]) / 2.0
    if centerBx > boxA[0] and centerBx < boxA[2] and centerBy > boxA[1] and centerBy < boxA[3]:
        return 1
    else:
        return 0

def generate_labels(boxA, boxB, OR_threshold):
    iou_value = ComputeIOU(boxA, boxB)
    return int(iou_value > OR_threshold)

def generate_pos(gt_box, block_size):
    block_centerx = np.random.randint(gt_box[0], gt_box[2], size=1, dtype=int)[0]
    block_centery = np.random.randint(gt_box[1], gt_box[3], size=1, dtype=int)[0]
    block_minx = int(block_centerx - block_size / 2)
    block_maxx = int(block_centerx + block_size / 2)
    block_miny = int(block_centery - block_size / 2)
    block_maxy = int(block_centery + block_size / 2)

    return (block_minx, block_maxx, block_miny, block_maxy)

def generate_neg(bboxes, well_minx, well_maxx, well_miny, well_maxy, block_size, iou_threshold = 0.05):
    success = False
    block_minx = 0
    block_maxx = 0
    block_miny = 0
    block_maxy = 0
    while(not success):
        block_centerx = np.random.randint(well_minx, well_maxx, size=1, dtype=int)[0]
        block_centery = np.random.randint(well_miny, well_maxy, size=1, dtype=int)[0]
        block_minx = int(block_centerx - block_size / 2)
        block_maxx = int(block_centerx + block_size / 2)
        block_miny = int(block_centery - block_size / 2)
        block_maxy = int(block_centery + block_size / 2)

        find_flag = True
        for bbox in bboxes:
            if ComputeOR(bbox, [block_minx, block_miny, block_maxx, block_maxy]) >= iou_threshold:
                find_flag = False
        success = find_flag

    return (block_minx, block_maxx, block_miny, block_maxy)


def generate_batch_data(images, gt_boxes, well_infos, resize, or_threshold, num, block_size = 24):
    """

    :param images:
    :param gt_boxes:
    :param well_infos:
    :param resize:
    :param or_threshold:
    :param num:
       :param block_size:
    :return:
        feature: N x C, N:samples, C:channels
        labels: N x 1
    """
    features = np.zeros((num, resize*resize), dtype=np.float32)
    labels = np.zeros(num, dtype=np.float32)

    label = 0

    for i in range(num):
        im = images[i]
        bboxes = gt_boxes[i]
        well_info = well_infos[i]  # well_centerx, well_centery, well_radius
        well_minx = well_info[0] - well_info[2]
        well_maxx = well_info[0] + well_info[2]
        well_miny = well_info[1] - well_info[2]
        well_maxy = well_info[1] + well_info[2]
        if label:
            block_minx, block_maxx, block_miny, block_maxy = generate_neg(bboxes = bboxes,
                                                                          well_minx = well_minx,
                                                                          well_maxx = well_maxx,
                                                                          well_miny = well_miny,
                                                                          well_maxy = well_maxy,
                                                                          block_size = block_size,
                                                                          iou_threshold=or_threshold)
            label = 0
        else:
            box_num = len(bboxes)
            if box_num<1:
                block_minx, block_maxx, block_miny, block_maxy = generate_neg(bboxes = bboxes,
                                                                              well_minx=well_minx,
                                                                              well_maxx=well_maxx,
                                                                              well_miny=well_miny,
                                                                              well_maxy=well_maxy,
                                                                              block_size=block_size,
                                                                              iou_threshold=or_threshold)
                label = 0
            else:
                box_ind = np.random.randint(0, box_num, size=1, dtype=int)[0]
                block_minx, block_maxx, block_miny, block_maxy = generate_pos(bboxes[box_ind], block_size)
                label = 1

        labels[i] = label
        #print(block_minx, block_maxx, block_miny, block_maxy, block_size)
        im_block = im[block_miny:block_maxy, block_minx:block_maxx]
        #if label == 0:
            ##cv2.imshow("im", im)
            #cv2.imshow("im_block", im_block)
            #cv2.waitKey(3000)
        im_block = cv2.resize(im_block, (resize, resize), fx=0, fy=0)
        feature = np.array(im_block, dtype=np.float32).reshape(1, -1)
        features[i, :] = feature[0, :] / 255.0
        #print(label)

    return features, labels
