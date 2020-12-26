import sys

from tensorflow_core.python.tools import freeze_graph

sys.path.append('../..')
from Methods.UNet_tf.util import *
import time
from datetime import timedelta
from Methods.UNet_tf.data import *
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

class UNet(object):

    def __init__(self, sess, conf):
        """
        本函数是UNet网络的初始化文件，用于构建网络结构、损失函数、优化方法。
        :param sess: tensorflow会话
        :param conf: 配置
        """
        self.sess = sess
        self.conf = conf
        # 输入图像
        self.images = tf.placeholder(tf.float32, shape=[None, self.conf.im_size, self.conf.im_size, 1], name='x')
        # 标注
        self.annotations = tf.placeholder(tf.float32, shape=[None, self.conf.im_size, self.conf.im_size, 2], name='annotations')
        # 构建UNet网络结构
        self.predict = self.inference()
        # 损失函数，分类精度
        self.loss_op = self.combined_loss()
        self.accuracy_op = self.accuracy()
        # 优化方法
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self.train_op()
        # 初始化参数
        self.sess.run(tf.global_variables_initializer())
        # 保存所有可训练的参数
        trainable_vars = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # 模型保存和保存summary的工具
        self.saver = tf.train.Saver(var_list=trainable_vars + bn_moving_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary('train')

    def config_summary(self, name):
        summarys = [tf.summary.scalar(name + '/loss', self.loss_op),
                    tf.summary.scalar(name + '/accuracy', self.accuracy_op)]
        summary = tf.summary.merge(summarys)
        return summary

    def save(self, step):
        print('saving', end=' ')
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)

        # Save check point for graph frozen later
        ckpt_filepath = self.save(directory=self.conf.modeldir, filename=self.conf.model_name)
        pbtxt_filename = self.conf.model_name + '.pbtxt'
        pbtxt_filepath = os.path.join(self.conf.modeldir, pbtxt_filename)
        pb_filepath = os.path.join(self.conf.modeldir, self.conf.model_name + '.pb')
        # This will only save the graph but the variables will not be saved.
        # You have to freeze your model first.


        self.saver.save(self.sess, ckpt_filepath, global_step=step)

        tf.train.write_graph(graph_or_graph_def=self.sess.graph_def, logdir=self.conf.modeldir, name=pbtxt_filename, as_text=True)

        # Freeze graph
        # Method 1
        freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False,
                                  input_checkpoint=ckpt_filepath, output_node_names='cnn/output',
                                  restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                                  output_graph=pb_filepath, clear_devices=True, initializer_nodes='')

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('------- no such checkpoint', model_path)
            return False
        self.saver.restore(self.sess, model_path)
        return True

    def save_summary(self, summary, step):
        print('summarizing', end=' ')
        self.writer.add_summary(summary, step)

    def inference(self):
        conv1 = conv(self.images, shape=[3, 3, 1, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv1 = conv(conv1, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool1 = max_pool(conv1, size=2)

        conv2 = conv(pool1, shape=[3, 3, 64, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv2 = conv(conv2, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool2 = max_pool(conv2, size=2)

        conv3 = conv(pool2, shape=[3, 3, 128, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv3 = conv(conv3, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool3 = max_pool(conv3, size=2)

        conv4 = conv(pool3, shape=[3, 3, 256, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv4 = conv(conv4, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool4 = max_pool(conv4, size=2)

        conv5 = conv(pool4, shape=[3, 3, 512, 1024], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv5 = conv(conv5, shape=[3, 3, 1024, 1024], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up6 = deconv(conv5, shape=[2, 2, 512, 1024], stride=2, stddev=0.1)
        merge6 = concat(up6, conv4)
        conv6 = conv(merge6, shape=[3, 3, 1024, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv6 = conv(conv6, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up7 = deconv(conv6, shape=[2, 2, 256, 512], stride=2, stddev=0.1)
        merge7 = concat(up7, conv3)
        conv7 = conv(merge7, shape=[3, 3, 512, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv7 = conv(conv7, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up8 = deconv(conv7, shape=[2, 2, 128, 256], stride=2, stddev=0.1)
        merge8 = concat(up8, conv2)
        conv8 = conv(merge8, shape=[3, 3, 256, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv8 = conv(conv8, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up9 = deconv(conv8, shape=[2, 2, 64, 128], stride=2, stddev=0.1)
        merge9 = concat(up9, conv1)
        conv9 = conv(merge9, shape=[3, 3, 128, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv9 = conv(conv9, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)

        predict = conv(conv9, shape=[3, 3, 64, 2], stddev=0.1, is_training=self.conf.is_training, stride=1, activation=False)

        return predict

    def loss(self, scope='loss'):
        """
        :return: 损失函数及分类精确度
        """
        # 标注 1通道-num_classes通道 one-hot
        with tf.variable_scope(scope):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.annotations, logits=self.predict)
            loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        return loss_op

    def binary_crossentropy(self, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
        Arguments:
          target: A tensor with the same shape as `output`.
          output: A tensor.
          from_logits: Whether `output` is expected to be a logits tensor.
              By default, we consider that `output`
              encodes a probability distribution.
        Returns:
          A tensor.
        """
        # Note: nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            epsilon_ = 1e-4 #_to_tensor(epsilon(), self.predict.dtype.base_dtype)
            output = clip_ops.clip_by_value(self.predict, epsilon_, 1 - epsilon_)
            output = math_ops.log(output / (1 - output))
            output = tf.cast(output, dtype=tf.float32)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=self.annotations, logits=output)

    def dice_loss(self, scope='dice_loss', smooth=1.):
        with tf.variable_scope(scope):
            y_true = tf.cast(self.annotations, tf.float32)
            y_pred = tf.math.sigmoid(self.predict)
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)
            loss_op = 1 - (numerator+smooth) / (denominator + smooth)

        return loss_op

    def combined_loss(self, scope='combine_loss', smooth=1.):
        with tf.variable_scope(scope):
            y_true = tf.cast(self.annotations, tf.float32)
            y_pred = tf.math.sigmoid(self.predict)
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)
            #numerator = 2 * tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=2), axis=2)
            #denominator = tf.reduce_sum(tf.reduce_sum(y_true, axis=2), axis=2) + tf.reduce_sum(tf.reduce_sum(y_pred, axis=2), axis=2)
            dice_loss = 1 - (numerator+smooth) / (denominator + smooth)
            dice_loss = tf.reduce_mean(dice_loss)
            dice_loss *= 1-self.conf.bce_weight

            #epsilon_ = 1e-4  # _to_tensor(epsilon(), self.predict.dtype.base_dtype)
            #output = clip_ops.clip_by_value(self.predict, epsilon_, 1 - epsilon_)
            #output = math_ops.log(output / (1 - output))
            #output = tf.cast(output, dtype=tf.float32)
            binary_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.annotations, logits=self.predict)
            binary_loss = tf.reduce_mean(binary_loss)
            binary_loss *= self.conf.bce_weight
            loss_op = dice_loss + binary_loss#tf.concat([dice_loss, binary_loss], axis=-1)
            #loss_op = tf.reduce_mean(loss_op)
        return loss_op

    def accuracy(self, scope='accuracy'):
        with tf.variable_scope(scope):
            preds = tf.cast(self.predict, dtype=tf.int64) #tf.argmax(self.predict, -1, name='accuracy/decode_pred')
            targets = tf.cast(self.annotations, dtype=tf.int64)
            acc = 1.0 - tf.nn.zero_fraction(
                tf.cast(tf.equal(preds, targets), dtype=tf.int32))
        return acc

    def train_op(self):
        # params = tf.trainable_variables()
        # gradients = tf.gradients(self.loss_op, params, name='gradients')
        # optimizer = tf.train.MomentumOptimizer(self.conf.rate, 0.9)
        # update = optimizer.apply_gradients(zip(gradients, params))
        # with tf.control_dependencies([update]):
        #     train_op = tf.no_op(name='train_op')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.rate).minimize(self.loss_op)
        return optimizer

    def train(self):
        if self.conf.reload_step > 0:
            if not self.reload(self.conf.reload_step):
                return
            print('reload', self.conf.reload_step)

        images, labels = read_record(self.conf.datadir, self.conf.im_size, self.conf.batch_size)
        with tf.device("/device:XLA_GPU:0"):
            tf.train.start_queue_runners(sess=self.sess)
            print('Begin Train')

            for train_step in range(1, self.conf.max_step + 1):
                start_time = time.time()

                x, y = self.sess.run([images, labels])
                # summary
                #print(x)
                #show = np.array(y[0, :, :, 1]*255, dtype = np.uint8)
                #cv2.imshow("show", show)
                #cv2.waitKey(0)
                if train_step == 1 or train_step % self.conf.summary_interval == 0:
                    feed_dict = {self.images: x,
                                 self.annotations: y}
                    loss, acc, _, summary = self.sess.run(
                        [self.loss_op, self.accuracy_op, self.optimizer, self.train_summary],
                        feed_dict=feed_dict)
                    print(str(train_step), '----Training loss:', loss, ' accuracy:', acc, end=' ')
                    self.save_summary(summary, train_step + self.conf.reload_step)
                    end_time = time.time()
                    time_diff = end_time - start_time
                    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
                # print 损失和准确性
                else:
                    feed_dict = {self.images: x,
                                 self.annotations: y}
                    loss, acc, _ = self.sess.run(
                        [self.loss_op, self.accuracy_op, self.optimizer], feed_dict=feed_dict)
                    #print(str(train_step), '----Training loss:', loss, ' accuracy:', acc, end=' ')
                # 保存模型
                if train_step % self.conf.save_interval == 0:
                    self.save(train_step + self.conf.reload_step)


    def predicts(self):
        if self.conf.reload_step > 0:
            if not self.reload(self.conf.reload_step):
                return
            print('reload', self.conf.reload_step)

        standard = self.images/255 - 0.5 #tf.image.per_image_standardization(self.images)

        for n in range(70, 100):
            img = os.path.join('data/render_test/Images/', str(n) + '.jpg')
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = np.reshape(img, (1, self.conf.im_size, self.conf.im_size, 1))

            img = self.sess.run([standard],
                                feed_dict={
                                    self.images: img
                                })
            img = np.reshape(img, (1, self.conf.im_size, self.conf.im_size, 1))

            label = self.sess.run([self.predict],
                                  feed_dict={
                                      self.images: img
                                  })
            label = np.array(label)
            label = label.reshape((self.conf.im_size, self.conf.im_size, 2))
            #label = np.argmax(label, axis=2)
            label = label[:, :, 1] * 255
            cv2.imshow("label", label)
            cv2.waitKey(0)
            print(n)
            #im = Image.fromarray(label.astype('uint8'))
            #im.save(os.path.join('data/render_test/predict/', str(n) + '.png'))