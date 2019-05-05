# coding=utf-8
import sklearn.metrics as metrics
import sklearn as sk
import tensorflow as tf
from cnn_model import CNN
from cnn_model import CNNConfig
import datetime
import time
import os


def train():
    # Training procedure
    # ======================================================
    # 设定最小显存使用量
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        config = CNNConfig()
        cnn = CNN(config)
        cnn.setVGG16()

        print('Setting Tensorboard and Saver...')
        # 设置Saver和checkpoint来保存模型
        # ===================================================
        checkpoint_dir = os.path.join(os.path.abspath("checkpoints"), "cnn")
        checkpoint_prefix = os.path.join(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # =====================================================

        # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        # ====================================================================
        train_tensorboard_dir = 'tensorboard/cnn/train/'
        test_tensorboard_dir = 'tensorboard/cnn/test/'
        if not os.path.exists(train_tensorboard_dir):
            os.makedirs(train_tensorboard_dir)
        if not os.path.exists(test_tensorboard_dir):
            os.makedirs(test_tensorboard_dir)

        # 训练结果记录
        log_file = open(test_tensorboard_dir+'/log.csv', mode='w', encoding='utf-8')
        log_file.write(','.join(['epoch', 'loss', 'precision', 'recall', 'f1_score']) + '\n')

        merged_summary = tf.summary.merge([tf.summary.scalar('loss', cnn.loss),
                                            tf.summary.scalar('accuracy', cnn.accuracy)])

        train_summary_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        # =========================================================================

        global_step = tf.Variable(0, trainable=False)
        # 衰减的学习率，每1000次衰减4%
        learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                   global_step, decay_steps=5000, decay_rate=0.98, staircase=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(cnn.loss, global_step)

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
            t = datetime.datetime.now().strftime('%m-%d %H:%M')
            print('%s: epoch: %d, step: %d, loss: %f, accuracy: %f' % (t, epoch, step, loss, accuracy))
            # 把结果写入Tensorboard中
            train_summary_writer.add_summary(summery, step)

        # 验证步骤
        def test_step(next_test_element):
            # 把test_loss和test_accuracy归0
            y_true = []
            y_pred = []
            test_loss = 0.0
            test_accuracy = 0.0
            test_precision = 0.0
            test_recall = 0.0
            test_f1_score = 0.0
            i = 0
            while True:
                try:
                    lines = sess.run(next_test_element)
                    batch_x, batch_y = cnn.convert_input(lines)
                    feed_dict = {
                        cnn.input_x: batch_x,
                        cnn.labels: batch_y,
                        cnn.dropout_keep_prob: 1.0,
                        cnn.training: False
                    }
                    # loss, pred, true = sess.run([cnn.loss, cnn.prediction, cnn.labels], feed_dict)
                    # 多次验证，取loss和score均值
                    mean_loss = 0
                    mean_score = 0
                    for i in range(config.multi_test_num):
                        loss, score = sess.run([cnn.loss, cnn.score], feed_dict)
                        mean_loss += loss
                        mean_score += score
                    mean_loss /= config.multi_test_num
                    mean_score /= config.multi_test_num
                    pred = sess.run(tf.argmax(mean_score, 1))
                    y_pred.extend(pred)
                    y_true.extend(batch_y)
                    test_loss += mean_loss
                    i += 1
                except tf.errors.OutOfRangeError:
                    # 遍历完验证集，计算评估
                    test_loss /= i
                    test_accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                    test_precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
                    test_recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
                    test_f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

                    t = datetime.datetime.now().strftime('%m-%d %H:%M')
                    log = '%s: epoch %d, testing loss: %0.6f, accuracy: %0.6f' % (
                        t, epoch, test_loss, test_accuracy)
                    log = log + '\n' + ('precision: %0.6f, recall: %0.6f, f1_score: %0.6f' % (
                        test_precision, test_recall, test_f1_score))
                    print(log)
                    log_file.write(','.join([str(epoch), str(test_loss),str(test_precision), str(test_recall),
                                             str(test_f1_score)]) + '\n')
                    time.sleep(3)
                    return

        print('Start training CNN...')
        sess.run(tf.global_variables_initializer())
        train_init_op, test_init_op, next_train_element, next_test_element = cnn.prepare_data()
        # Training loop
        for epoch in range(config.epoch_num):
            sess.run(train_init_op)
            while True:
                try:
                    lines = sess.run(next_train_element)
                    batch_x, batch_y = cnn.convert_input(lines)
                    train_step(batch_x, batch_y, config.dropout_keep_prob)
                except tf.errors.OutOfRangeError:
                    # 初始化验证集迭代器
                    sess.run(test_init_op)
                    # 计算验证集准确率
                    test_step(next_test_element)
                    break
        train_summary_writer.close()
        log_file.close()
        # 训练完成后保存参数
        path = saver.save(sess, checkpoint_prefix, global_step=global_step)
        print("Saved model checkpoint to {}\n".format(path))
    # ==================================================================


if __name__ == '__main__':
    train()






