import tensorflow as tf
from tensorflow.data.experimental import CsvDataset
from tensorflow.data import TextLineDataset
import numpy as np
from fer2013 import preprocess
import os

class CNNConfig(object):
    """
    # TODO: 在此修改TextCNN以及训练的参数
    """

    class_num = 7        # 输出类别的数目
    img_size = 48          # 图像的尺寸

    dropout_keep_prob = 0.5     # dropout保留比例（弃用）
    learning_rate = 1e-3    # 学习率
    train_batch_size = 128         # 每批训练大小
    test_batch_size = 500        # 每批测试大小
    test_per_batch = 250           # 每多少批进行一次测试
    epoch_num = 30001        # 总迭代轮次


class CNN(object):
    def __init__(self, config):
        self.class_num = config.class_num
        self.img_size = config.img_size
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.test_per_batch = config.test_per_batch

        self.batch_x = ''
        self.batch_y = ''
        self.input_x = ''
        self.input_y = ''
        self.dropout_keep_prob = ''
        self.training = ''
        self.embedding_inputs = ''
        self.embedding_inputs_expanded = ''
        self.loss = ''
        self.accuracy = ''
        self.prediction = ''
        self.vocab = ''

    def setCNN(self):
        """
        在此函数中设定CNN模型
        :return:
        """
        def filter_variable(shape):
            # 通过Truncated normal distribution（截断正态分布）生成随机数tensor
            init = tf.truncated_normal(shape=shape, stddev=0.1)
            return tf.Variable(init)

        def weight_variable(shape):
            # 通过Truncated normal distribution（截断正态分布）生成随机数tensor
            init = tf.truncated_normal(shape=shape, stddev=0.1)
            return tf.Variable(init)

        def bias_variable(shape):
            init = tf.constant(0.1, shape=shape)
            return tf.Variable(init)

        def conv2d(input, filter):
            """
            tf.nn.conv2d()创建一个处理2维的卷积层\n
            input是输入的数据，shape=[batch, in_height, in_width, in_channels]\n
            filter是卷积核的参数，也即卷积层的参数，
            shape=[filter_height, filter_widt,h in_channels, out_channels]\n
            in_channels即输入的通道数，黑白为1，RGB为3\n
            out_channels即卷积层的深度（卷积核个数）\n
            stirde[1, horizontal, vertical, 1]是步长\n
            padding表示卷积的方式，"SAME"通过填充图像使卷积后的输出和原图像大小一致，
            "VALID"则不填充，卷积剩下的像素直接舍弃
            """
            return tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding="VALID")

        def max_pool_3x3_2(input):
            """
            x是输入的数据\n
            ksize是池化窗口的大小[1, height, width, 1]\n
            stride同conv2d的stride\n
            """
            return tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        def max_pool_5x5_1(input):
            return tf.nn.max_pool(input, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding="SAME")

        def max_pool_2x2_2(input):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Input layer
        self.input_x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1], name="input_x")
        self.labels = tf.placeholder(tf.int32, [None], name="input_y")
        self.input_y = tf.one_hot(self.labels, self.class_num)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

        # 1_conv
        filter1 = filter_variable(shape=[5, 5, 1, 32])
        b1 = bias_variable(shape=[32])
        output_1_conv = tf.nn.relu(conv2d(input=self.input_x, filter=filter1) + b1)  # output 44*44*32
        output_1_conv = tf.layers.batch_normalization(output_1_conv, training=self.training)
        # 2_max_pool
        output_2_max_pool = max_pool_3x3_2(output_1_conv)  # output 22*22*32
        # 3_conv
        filter3 = filter_variable(shape=[5, 5, 32, 64])
        b3 = bias_variable(shape=[64])
        output_3_conv = tf.nn.relu(conv2d(input=output_2_max_pool, filter=filter3) + b3)
        # output 18*18*64
        # 4_max_pool
        output_4_max_pool = max_pool_5x5_1(input=output_3_conv)  # output 18*18*64
        # 5_conv
        filter5 = filter_variable(shape=[4, 4, 64, 128])
        b5 = bias_variable(shape=[128])
        output_5_conv = tf.nn.relu(conv2d(input=output_4_max_pool, filter=filter5) + b5)  # output 15*15*128
        # 6_fc with 3072 neurons
        W6 = weight_variable(shape=[15 * 15 * 128, 2048])
        b6 = bias_variable(shape=[2048])
        output_5_conv_flat = tf.reshape(output_5_conv, shape=[-1, 15 * 15 * 128])
        output_6_fc = tf.nn.relu(tf.matmul(output_5_conv_flat, W6) + b6)
        # output -1*2048
        # 7_fc with 7 neurons
        W7 = weight_variable(shape=[2048, self.class_num])
        b7 = bias_variable(shape=[self.class_num])
        output_7_fc = tf.matmul(output_6_fc, W7) + b7

        self.prediction = tf.argmax(output_7_fc, 1, name='prediction')

        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_7_fc, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def convert_input(self, pixels, labels):
        """
        把读取的字符串数据转为形状为[batch_size, img_size, img_size, 1]的数组，
        作为CNN的输入

        :param pixels:
        :param labels:
        :return:
        """
        batch_x = []
        for row in pixels:
            batch_x.append([int(x) for x in row.decode('utf-8').split()])
        batch_x = np.stack(batch_x)
        batch_x = batch_x.reshape([-1, self.img_size, self.img_size, 1])
        batch_y = labels

        return batch_x, batch_y

    def prepare_train_data(self):
        # Data preparation.
        # =======================================================
        train_dataset = CsvDataset(os.path.join('fer2013', preprocess.valid_train_data_path),
        [tf.int32, tf.string], header=True, select_cols=[0,1]).shuffle(preprocess.valid_train_size)
        train_dataset = train_dataset.batch(self.train_batch_size).repeat()

        train_iterator = train_dataset.make_initializable_iterator()

        train_init_op = train_iterator.initializer

        next_train_element = train_iterator.get_next()

        return train_dataset, train_init_op, next_train_element
        # =============================================================
        # Date preparation ends.

    def prepare_test_data(self):
        test_dataset = CsvDataset(os.path.join('fer2013', preprocess.valid_test_data_path),
        [tf.int32, tf.string], header=True, select_cols=[0,1]).shuffle(preprocess.valid_test_size)

        test_dataset = test_dataset.batch(self.test_batch_size).repeat()
        test_iterator = test_dataset.make_initializable_iterator()
        test_init_op = test_iterator.initializer
        next_test_element = test_iterator.get_next()
        return test_dataset, test_init_op, next_test_element





