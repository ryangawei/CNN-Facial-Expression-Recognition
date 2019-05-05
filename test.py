# coding=utf-8
import sklearn.metrics as metrics
import sklearn as sk
from tensorflow.data import TextLineDataset
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_model import CNN
from cnn_model import CNNConfig
import datetime
import time
import os
from data import preprocess
import numpy as np
import cv2


class Predictor(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.class_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def init_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.config = CNNConfig()
        self.cnn = CNN(self.config)
        # self.cnn.setVGG16()

        print('Loading model from file:', self.model_path)
        saver = tf.train.import_meta_graph(self.model_path + '.meta')
        saver.restore(self.sess, self.model_path)
        self.graph = tf.get_default_graph()
        # 从图中读取变量
        self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        self.labels = self.graph.get_operation_by_name("labels").outputs[0]
        self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        self.score = self.graph.get_operation_by_name('score/Relu').outputs[0]
        self.prediction = self.graph.get_operation_by_name("prediction").outputs[0]
        self.training = self.graph.get_operation_by_name("training").outputs[0]

    def predict(self, batch_x):
        feed_dict = {
            self.input_x: batch_x,
            self.dropout_keep_prob: 1.0,
            self.training: False
        }
        score, pre = self.sess.run([self.score, self.prediction], feed_dict)
        return score, pre

    def draw_confusion_matrix(self):
        # train_init_op, test_init_op, next_train_element, next_test_element = self.cnn.prepare_data()
        test_dataset = TextLineDataset(os.path.join('data', preprocess.FILTERED_TEST_PATH)).skip(1).batch(self.cnn.test_batch_size)
        # Create a reinitializable iterator
        test_iterator = test_dataset.make_one_shot_iterator()
        next_test_element = test_iterator.get_next()

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
                lines = self.sess.run(next_test_element)
                batch_x, batch_y = self.cnn.convert_input(lines)
                feed_dict = {
                    self.input_x: batch_x,
                    self.labels: batch_y,
                    self.dropout_keep_prob: 1.0,
                    self.training: False
                }
                # loss, pred, true = sess.run([self.cnn.loss, self.cnn.prediction, self.cnn.labels], feed_dict)
                # 多次验证，取loss和score均值
                mean_score = 0
                for i in range(self.config.multi_test_num):
                    score = self.sess.run(self.score, feed_dict)
                    mean_score += score
                mean_score /= self.config.multi_test_num
                pred = self.sess.run(tf.argmax(mean_score, 1))
                y_pred.extend(pred)
                y_true.extend(batch_y)
                i += 1
            except tf.errors.OutOfRangeError:
                # 遍历完验证集，计算评估
                test_loss /= i
                test_accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                test_precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
                test_recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
                test_f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
                log =('precision: %0.6f, recall: %0.6f, f1_score: %0.6f' % (
                    test_precision, test_recall, test_f1_score))
                print(log)

                cm = confusion_matrix(y_true, y_pred)
                print('Total samples:', np.sum(cm))
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
                print('Confusion matrix:\n', cm)
                # 绘制混淆矩阵
                # ==============================================================
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                # We want to show all ticks...
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       # ... and label them with the respective list entries
                       xticklabels=self.class_name, yticklabels=self.class_name,
                       title="Normalized confusion matrix",
                       ylabel='True label',
                       xlabel='Predicted label')

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                fmt = '.2f'
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], fmt),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                fig.tight_layout()
                plt.savefig('./data/confusion_matrix.jpg')
                plt.show()
                # =====================================================================
                break

    def _detect_sentiment(self, detector, img):
        # 转为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            image=gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags=0
        )
        if len(faces) != 0:
            batch_x = []
            for face in faces:
                x, y, w, h = face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # opencv的图像是[y, x]储存的
                # 裁剪并显示人脸部分
                img_cropped = cv2.resize(gray[y:y + h, x:x + w], (self.cnn.img_size, self.cnn.img_size))
                cv2.imshow('cropped', img_cropped)
                img_input = img_cropped.reshape([self.cnn.img_size, self.cnn.img_size, 1])
                batch_x.append(img_input)
            batch_x = np.stack(batch_x)
            mean_score = 0
            for i in range(self.config.multi_test_num):
                score, _ = self.predict(batch_x)
                mean_score += score
            mean_score /= self.config.multi_test_num
            pred = self.sess.run(tf.argmax(mean_score, 1))
            for i in range(len(faces)):
                # 给score显示条形图
                # =======================================================
                plt.bar(range(self.cnn.class_num), mean_score[i], align='center', color='steelblue', alpha=0.8)
                plt.ylabel('Score')
                plt.xticks(range(self.cnn.class_num), self.class_name)
                plt.show()
                # ========================================================
                cv2.putText(img=img, text=self.class_name[pred[i]],
                            org=(faces[i][0], faces[i][1] + faces[i][3] + 20),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.6,
                            color=(0, 0, 255))
            return img
        else:
            return None

    def camera_detect(self):
        # 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
        while True:
            # 从摄像头读取图片
            sucess, img = cam.read()
            if not sucess:
                continue
            img = self._detect_sentiment(detector, img)
            # 显示摄像头，背景是灰度。
            if img is not None:
                cv2.imshow("Sentiment Detection", img)
            # 保持画面的持续。
            k = cv2.waitKey(1)
            if k == 27:
                # 通过esc键退出摄像
                cv2.destroyAllWindows()
                break
            elif k == ord("s"):
                # 通过s键保存图片，并退出。
                cv2.imwrite("capture.jpg", img)
                cv2.destroyAllWindows()
                break
        # 关闭摄像头
        cam.release()

    def image_detect(self, img_path):
        img = cv2.imread(img_path)
        detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
        img = self._detect_sentiment(detector, img)
        cv2.imshow('Sentiment Detection Result', img)
        k = cv2.waitKey()
        if k == ord("s"):
            # 通过s键保存图片，并退出。
            cv2.imwrite("result.jpg", img)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    predictor = Predictor('./checkpoints/cnn-20700')
    predictor.init_model()
    predictor.draw_confusion_matrix()
    # predictor.camera_detect()
    # predictor.image_detect('./data/sad2.jpg')

