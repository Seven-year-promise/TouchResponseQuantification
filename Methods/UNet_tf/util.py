import tensorflow as tf
import cv2
import numpy as np

def well_detection(im, gray, threshold = 50):
    # gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 5,
                               param1=220, param2=30,
                               minRadius=95, maxRadius=105)
    #print(circles)
    '''
    #muted when training
    if circles is not None:
        circles_int = np.uint16(np.around(circles))
        for i in circles_int[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(gray, center, 1, (0, 255, 0), 3)
            # circle outline
            radius = i[2]
            cv2.circle(gray, center, radius, (0, 255, 0), 3)

    cv2.imshow("detected circles", gray)
    cv2.waitKey(1000)
    '''
    if circles is not None:
        well_centerx = np.uint16(np.round(np.average(circles[0, :, 0])))
        well_centery = np.uint16(np.round(np.average(circles[0, :, 1])))
        well_radius = 110 #np.uint16(np.round(np.max(circles[0, :, 2])))
        #return True, (well_centerx, well_centery, 110)


    else:
        well_centerx = 240
        well_centery = 240
        well_radius = 110
        #return False, (240, 240, 110)

    # first rough mask for well detection
    mask = np.zeros(gray.shape[:2], dtype="uint8")
    cv2.circle(mask, (well_centerx, well_centery), well_radius, 255, -1)

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_masked_color = cv2.cvtColor(gray_masked, cv2.COLOR_GRAY2BGR)
    '''
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
    #closing_inv = np.array((closing_inv, closing_inv, closing_inv)).transpose(1, 2, 0)
    closing_inv = cv2.cvtColor(closing_inv, cv2.COLOR_GRAY2BGR)
    im_closing_inv = closing_inv + im_closing
    #cv2.circle(gray, (well_centerx, well_centery), 1, (0, 255, 0), 5)
    #cv2.imshow("detected circles", im_closing_inv)
    #cv2.waitKey(1000)
    '''
    return True, (well_centerx, well_centery, well_radius), gray_masked_color

def well_detection_strong(im, gray, threshold = 50):
    """
    only difference for the kernel = np.ones((20, 20), dtype=np.uint8) to delete more dark edge
    """
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

# Weights
def new_weights(shape, stddev):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev))


# Biases
def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))


# Convolutional layer
def conv(input, shape, stddev, is_training, stride, name=None, activation=True):
    """
    :param stride: stride
    :param is_training: 是否训练
    :param input: 输入
    :param shape: 过滤器尺寸
    :param stddev: 初始化
    :param activation: activation function
    :return:
    """

    weights = new_weights(shape, stddev)
    biases = new_biases(shape[-1])
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
    layer = tf.math.add(layer, biases, name=name)
    #layer = batch_normalization(layer, training=is_training)
    if activation:
        layer = tf.nn.relu(layer)
    return layer

def MFM(input, name):
    with tf.variable_scope(name):
        #shape is in format [batchsize, x, y, channel]
        # shape = tf.shape(x)
        shape = input.get_shape().as_list()
        res = tf.reshape(input,[-1,shape[1],shape[2],2,shape[-1]//2])
        # x2 = tf.reshape(x,[-1,2,shape[1]//2, shape[2], shape[3]])
        res = tf.reduce_max(res, axis=[3])
        # x2 = tf.reduce_max(x2,axis=[1])
        # x3 = tf.reshape(x2,[-1,int(x2.get_shape()[3]), int(x2.get_shape()[2]), int(x2.get_shape()[1])])
        return res

def deconv(input, shape, stride, stddev):
    in_shape = tf.shape(input)
    output_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] // 2])
    weights = new_weights(shape, stddev)
    return tf.nn.conv2d_transpose(input, filter=weights, output_shape=output_shape,
                                  strides=[1, stride, stride, 1], padding='SAME')


def max_pool(input, size):
    return tf.nn.max_pool2d(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def concat(a, b):
    return tf.concat([a, b], 3)


def batch_normalization(inputs, training):
    bn = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=0.997,
        epsilon=1e-5,
        center=True,
        scale=True,
        training=training,
        fused=True)

    return bn