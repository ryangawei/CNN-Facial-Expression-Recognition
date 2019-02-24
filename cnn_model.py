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
    epoch_num = 25001        # 总迭代轮次


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

    def setCNN(self):
        """
        在此函数中设定CNN模型
        参考subnet1, Tabel I
        from https://ieeexplore.ieee.org/document/7756145
        """

        # Input layer
        self.input_x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1], name="input_x")
        self.labels = tf.placeholder(tf.int32, [None], name="input_y")
        self.input_y = tf.one_hot(self.labels, self.class_num)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

        # 1_卷积层
        output_1_conv = tf.layers.conv2d(
            inputs=self.input_x,
            filters=64,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1),
        )
        # 2_池化层
        output_2_max_pool = tf.layers.max_pooling2d(
            inputs=output_1_conv,
            pool_size=[2, 2],
            strides=[2, 2]
        )
        # 3_卷积层
        output_3_conv = tf.layers.conv2d(
            inputs=output_2_max_pool,
            filters=128,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1),
        )
        # 4_池化层
        output_4_max_pool = tf.layers.max_pooling2d(
            inputs=output_3_conv,
            pool_size=[2, 2],
            strides=[2, 2]
        )
        # 5_卷积层
        output_5_conv = tf.layers.conv2d(
            inputs=output_4_max_pool,
            filters=256,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1)
        )
        # 6_池化层
        output_6_max_pool = tf.layers.max_pooling2d(
            inputs=output_5_conv,
            pool_size=[2, 2],
            strides=[2, 2]
        )
        # 输出形状为[-1, 6, 6, 256]
        output_6_max_pool = tf.reshape(output_6_max_pool, [-1, 9216])
        # 7_全连接层
        output_7_fc = tf.layers.dense(
            inputs=output_6_max_pool,
            units=4096,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1)
        )
        # 8_全连接层
        output_8_fc = tf.layers.dense(
            inputs=output_7_fc,
            units=4096,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1)
        )
        # 9_输出层
        score = tf.layers.dense(
            inputs=output_8_fc,
            units=self.class_num,
            activation=None,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1)
        )

        self.prediction = tf.argmax(score, 1, name='prediction')

        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=self.input_y)
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





