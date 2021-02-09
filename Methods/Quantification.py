from Methods.UNet_tf.test import *

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

def find_next_point(larva_centers, this_binary):


def larva_tracking(video, model_path):
    unet_test = UNetTestTF()
    unet_test.model.load_graph_frozen(model_path=model_path)
    i = 0
    last_binary, this_binary = None, None

    larva_centers = []
    for im in video:
        i += 1
        unet_test.load_im(im)
        _, this_binary, larva_blobs = unet_test.predict(threshold=0.9, size=44)

        if last_binary is None:
            for b in larva_blobs:
                center = np.round(np.average(b, axis=1))
                larva_centers.append([int(center[0]), int(center[1])])
        else:

        last_binary = this_binary

if __name__ == '__main__':
    video_path = ""
    video = []
    larva_tracking(video, model_path="Methods/UNet_tf/LightCNN/models_rotate_contrast/UNet60000.pb")