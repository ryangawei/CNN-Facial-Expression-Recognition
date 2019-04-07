import tensorflow as tf
from tensorflow.data import TextLineDataset
import numpy as np
from data import preprocess
import os
import random


class CNNConfig(object):
    """
    # TODO: 在此修改TextCNN以及训练的参数
    """
    def __init__(self):
        self.class_num = 7        # 输出类别的数目
        self.img_size = 48          # 图像的尺寸
        self.crop_size = 43          # 数据增强时，随机裁剪的尺寸
        self.dropout_keep_prob = 0.5     # dropout保留比例（弃用）
        self.learning_rate = 1e-4   # 学习率
        self.train_batch_size = 128         # 每批训练大小
        self.test_batch_size = 500        # 每批测试大小
        self.test_per_batch = 250           # 每多少批进行一次测试
        self.epoch_num = 25        # 总迭代轮次


class CNN(object):
    def __init__(self, config):
        self.class_num = config.class_num
        self.img_size = config.img_size
        self.crop_size = config.crop_size
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.test_per_batch = config.test_per_batch

        self.batch_x = ''
        self.batch_y = ''
        self.input_x = ''
        self.labels = ''
        self.input_y = ''
        self.dropout_keep_prob = ''
        self.training = ''
        self.embedding_inputs = ''
        self.embedding_inputs_expanded = ''
        self.loss = ''
        self.accuracy = ''
        self.prediction = ''
        self.vocab = ''

    def _set_input(self):
        """
        在此函数中设定CNN模型
        参考subnet1, Tabel I
        from https://ieeexplore.ieee.org/document/7756145
        """

        # Input layer
        self.input_x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1], name="input_x")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        self.input_y = tf.one_hot(self.labels, self.class_num, name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

    def setVGG16(self):
        self._set_input()
        self.input_x_enhanced = self.data_enhance(self.input_x)

        def _conv(input, ksize, stride, filters):
            return tf.layers.conv2d(
                inputs=input,
                filters=filters,
                kernel_size=[ksize, ksize],
                strides=[stride, stride],
                padding='SAME',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                bias_initializer=tf.initializers.constant(0.0),
            )

        def _maxpool_2x2(input):
            return tf.layers.max_pooling2d(
                inputs=input,
                pool_size=[2, 2],
                strides=[2, 2],
                padding='SAME',
            )

        def _fc(input, units, dropout_keep_prob=self.dropout_keep_prob):
            fc_output = tf.layers.dense(
            inputs=input,
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.initializers.constant(0.0)
            )
            return tf.layers.dropout(fc_output, dropout_keep_prob)

        # 加入批标准化以减少过拟合
        input_x_norm = tf.layers.batch_normalization(self.input_x_enhanced, training=self.training)
        # conv3-64
        conv3_64_1 = _conv(input_x_norm, 3, 1, 64)
        conv3_64_output = _conv(conv3_64_1, 3, 1, 64)

        # maxpool-1
        maxpool_1_output = _maxpool_2x2(conv3_64_output)

        # conv3-128
        conv3_128_1 = _conv(maxpool_1_output, 3, 1, 128)
        conv3_128_output = _conv(conv3_128_1, 3, 1, 128)

        # maxpool-2
        maxpool_2_output = _maxpool_2x2(conv3_128_output)

        # conv3-256
        conv3_256_1 = _conv(maxpool_2_output, 3, 1, 256)
        conv3_256_2 = _conv(conv3_256_1, 3, 1, 256)
        conv3_256_output = _conv(conv3_256_2, 3, 1, 256)

        # maxpool-3
        maxpool_3_output = _maxpool_2x2(conv3_256_output)

        # conv3-512
        conv3_512_1 = _conv(maxpool_3_output, 3, 1, 512)
        conv3_512_2 = _conv(conv3_512_1, 3, 1, 512)
        conv3_512_output = _conv(conv3_512_2, 3, 1, 512)

        # maxpool-4
        maxpool_4_output = _maxpool_2x2(conv3_512_output)

        # conv4-512
        conv4_512_1 = _conv(maxpool_4_output, 3, 1, 512)
        conv4_512_2 = _conv(conv4_512_1, 3, 1, 512)
        conv4_512_output = _conv(conv4_512_2, 3, 1, 512)

        # maxpool-5
        maxpool_5_output = _maxpool_2x2(conv4_512_output)

        # flatten
        shape = maxpool_5_output.shape.as_list()
        dims = shape[1]*shape[2]*shape[3]
        maxpool_5_output_flatten = tf.reshape(maxpool_5_output, [-1, dims])

        # fully-connected-1
        fc_1 = _fc(maxpool_5_output_flatten, 4096)

        # fully-connected-2
        fc_2 = _fc(fc_1, 4096)

        # fully-connected-3
        fc_3 = _fc(fc_2, 1000)

        # 输出层
        score = _fc(fc_3, self.class_num)

        self.prediction = tf.argmax(score, 1, name='prediction')

        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def convert_input(self, lines):
        """
        把读取的字符串数据转为形状为[batch_size, img_size, img_size, 1]的数组，
        作为CNN的输入
        :param pixels:
        :param labels:
        :return:
        """
        batch_x = []
        batch_y = []
        for line in lines:
            line_ = line.decode('utf-8').strip().split(',')
            pixels = line_[1].split()  # 像素值
            label = line_[0]  # 第一项为标签
            batch_x.append([int(x) for x in pixels])
            batch_y.append(int(label))
        batch_x = np.stack(batch_x)
        batch_x = batch_x.reshape([-1, self.img_size, self.img_size, 1])

        return batch_x, batch_y

    def prepare_data(self):
        self.train_dataset = TextLineDataset(os.path.join('data', preprocess.TRAIN_PATH)).skip(1)
        self.test_dataset = TextLineDataset(os.path.join('data', preprocess.TEST_PATH)).skip(1)

    def shuffle_datset(self):
        # 打乱数据集
        # ==========================================================
        print('Shuffling dataset...')
        # 打乱数据
        train_dataset = self.train_dataset.batch(self.train_batch_size)
        test_dataset = self.test_dataset.batch(self.test_batch_size)

        # Create a reinitializable iterator
        train_iterator = train_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()

        train_init_op = train_iterator.initializer
        test_init_op = test_iterator.initializer

        # 要获取元素，先sess.run(train_init_op)初始化迭代器
        # 再sess.run(next_train_element)
        next_train_element = train_iterator.get_next()
        next_test_element = test_iterator.get_next()

        return train_init_op, test_init_op, next_train_element, next_test_element
        # ==============================================================

    def data_enhance(self, images):
        # 对图片进行数据增强
        # 随机水平翻转
        # images = tf.image.random_flip_left_right(images)
        # 随机裁剪
        # images = tf.image.random_crop(images, [self.train_batch_size ,self.crop_size, self.crop_size, 1])
        # images = tf.image.random_saturation(images, 0.5, 1.5)
        # images = tf.image.random_contrast(images, 0.5, 1.5)
        return images





