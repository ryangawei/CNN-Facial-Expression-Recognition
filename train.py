import tensorflow as tf
from cnn_model import CNN
from cnn_model import CNNConfig
import os
import datetime
import time


def train():
    # Training procedure
    # ======================================================
    with tf.Session() as sess:
        config = CNNConfig()
        cnn = CNN(config)
        train_dataset, train_init_op, next_train_element = cnn.prepare_train_data()
        test_dataset, test_init_op, next_test_element = cnn.prepare_test_data()
        cnn.setCNN()

        print('Setting Tensorboard and Saver...')
        # 设置Saver和checkpoint来保存模型
        # ===================================================
        checkpoint_dir = os.path.abspath("checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # =====================================================

        # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        # ====================================================================
        train_tensorboard_dir = 'tensorboard/cnn/train'
        test_tensorboard_dir = 'tensorboard/cnn/test'
        if not os.path.exists(train_tensorboard_dir):
            os.makedirs(train_tensorboard_dir)
        if not os.path.exists(test_tensorboard_dir):
            os.makedirs(test_tensorboard_dir)
        tf.summary.scalar('loss', cnn.loss)
        tf.summary.scalar('accuracy', cnn.accuracy)
        merged_summary = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        valid_summary_writer = tf.summary.FileWriter(test_tensorboard_dir, sess.graph)
        # =========================================================================

        global_step = tf.Variable(0, trainable=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(cnn.loss, global_step)

        # 训练步骤
        def train_step(batch_x, batch_y, keep_prob=config.dropout_keep_prob):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: keep_prob,
                cnn.training: True
            }
            sess.run(train_op, feed_dict=feed_dict)
            step, loss, accuracy, summery = sess.run(
                [global_step, cnn.loss, cnn.accuracy, merged_summary],
                feed_dict={cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0,
                cnn.training: False})
            time = datetime.datetime.now().isoformat()
            print('%s: step: %d, loss: %f, accuracy: %f' % (time, step, loss, accuracy))
            # 把结果写入Tensorboard中
            train_summary_writer.add_summary(summery, step)

        # 测试步骤
        def valid_step(batch_x, batch_y):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0,
                cnn.training: False
            }

            total_loss = 0.0
            total_accuracy = 0.0
            # 验证10次取平均值
            for _ in range(10):
                loss, accuracy, summery = sess.run([cnn.loss, cnn.accuracy, merged_summary],
                                            feed_dict)
                total_loss += loss
                total_accuracy += accuracy
            total_loss /= 10
            total_accuracy /= 10

            print('Test loss: %f, accuracy: %f' % (total_loss, total_accuracy))
            # 把结果写入Tensorboard中
            valid_summary_writer.add_summary(summery)

        sess.run(tf.global_variables_initializer())

        # 初始化训练集、验证集迭代器
        sess.run(train_init_op)
        sess.run(test_init_op)

        # Training loop
        for epoch in range(config.epoch_num):
            labels, pixels = sess.run(next_train_element)
            batch_x, batch_y = cnn.convert_input(pixels, labels)
            train_step(batch_x, batch_y, config.dropout_keep_prob)
            if epoch % config.test_per_batch == 0:
                labels, pixels = sess.run(next_test_element)
                batch_x, batch_y = cnn.convert_input(pixels, labels)
                valid_step(batch_x, batch_y)
                time.sleep(3)

        train_summary_writer.close()
        valid_summary_writer.close()

        # 训练完成后保存参数
        path = saver.save(sess, checkpoint_prefix, global_step=global_step)
        print("Saved model checkpoint to {}\n".format(path))
    # ==================================================================


if __name__ == '__main__':
    train()







