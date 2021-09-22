import tensorflow as tf
from PIL import Image
import os
import numpy as np
import cv2

FILE_NAME = [""] #,"01202","01203","01204","01205"]
def create_record(data_path, im_size, records_path):
    writer = tf.io.TFRecordWriter(records_path)
    base_im_path = data_path + 'Images/'
    base_anno_path = data_path + "annotation/"
    for fileN in FILE_NAME:
        train_im_path = base_im_path + fileN + "/"
        train_anno_path = base_anno_path + fileN + "/"
        ims_name = os.listdir(train_im_path)
        annos_name = os.listdir(train_anno_path)
        im_anno_list = []
        for im_name in ims_name:
            name = im_name[:-4]
            img = cv2.imread(train_im_path + im_name)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            '''
            anno = cv2.imread(train_anno_path + name + "_label.tif")
            # anno = cv2.erode(anno, (3, 3), iterations=2)
            anno = anno[:, :, 1]
            anno_needle = np.zeros((im_size, im_size), dtype=np.uint8)
            anno_needle[np.where(anno == 1)] = 1
            anno_fish = np.zeros(anno.shape, dtype=np.uint8)
            anno_fish[np.where(anno == 2)] = 1
            '''
            anno_needle = cv2.imread(train_anno_path + name + "_label_1.tif")
            # anno = cv2.erode(anno, (3, 3), iterations=2)
            anno_needle = anno_needle[:, :, 1]

            anno_fish = cv2.imread(train_anno_path + name + "_label_2.tif")
            # anno = cv2.erode(anno, (3, 3), iterations=2)
            anno_fish = anno_fish[:, :, 1]

            anno_together = np.zeros((im_size, im_size, 2), dtype=np.uint8)
            anno_together[:, :, 0] = anno_needle
            anno_together[:, :, 1] = anno_fish

            img_raw = img.tobytes()
            label_raw = anno_together.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(example.SerializeToString())
    writer.close()


def read_record(filename, im_size, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'img': tf.FixedLenFeature([], tf.string),
        }
    )
    img = features['img']
    label = features['label']

    img = tf.decode_raw(img, tf.uint8)
    label = tf.decode_raw(label, tf.uint8)

    img = tf.cast(img, dtype=tf.int32)
    label = tf.cast(label, dtype=tf.float32)

    img = tf.reshape(img, [im_size, im_size, 3])
    label = tf.reshape(label, [im_size, im_size, 2])

    #data = tf.concat([img, label], axis=2)

    #data = tf.image.random_flip_left_right(data)
    #data = tf.image.random_flip_up_down(data)

    #data = tf.transpose(data, [2, 0, 1])
    #img = data[0]
    #label = data[1]

    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [im_size, im_size, 3])
    #img = tf.image.per_image_standardization(img)
    #img = img / 255 - 0.5

    min_after_dequeue = 30
    capacity = min_after_dequeue + 3 * batch_size
    img, label = tf.train.shuffle_batch([img, label],
                                        batch_size=batch_size,
                                        num_threads=6,
                                        capacity=capacity,
                                        min_after_dequeue=min_after_dequeue)
    return img, label


if __name__ == '__main__':
    if not os.path.exists('./data/train/train.tfrecords'):
        create_record('data/train/', 480, './data/train/train.tfrecords')
    else:
        print('TFRecords already exists!')