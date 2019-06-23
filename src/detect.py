# coding=utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
from config import *
from preprocess import *
import argparse
import keras.backend as K
import keras
from keras.backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os


class Detector():
    def __init__(self, **kwargs):
        c = K.tf.ConfigProto()
        c.gpu_options.allow_growth = True
        session = K.tf.Session(config=c)
        set_session(session)

        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)
        self.tta_datagen = ImageDataGenerator(
                        samplewise_center=True,
                        samplewise_std_normalization=True,
                        brightness_range=(0.8, 1.2),
                        rotation_range=10,
                        width_shift_range=0.10,
                        height_shift_range=0.10,
                        zoom_range=0.10,
                        horizontal_flip=True,
                        )
        self.datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
        )

        model_save_path = MODEL_DIR + '/' + 'model_{}.h5'.format(kwargs['model'])
        self.model = keras.models.load_model(model_save_path)
        self.tta = kwargs['tta']
        self.cam = kwargs['cam']
        self.img_path = kwargs['path']

    def _detect_sentiment(self, img, show_score=False):
        """
        检测图片中人脸的表情，返回标记了人脸及表情的图片

        :param detector:
        :param img:
        :param show_score: 是否用条形图显示脸部的score
        :return:
        """
        # 转为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_faces, all_landmarks = crop_face_area(self.detector, self.landmark_predictor, gray, img_size)
        if all_faces is not None:
            all_landmarks = all_landmarks.astype(np.int32)
            batch_x = np.expand_dims(all_faces, -1)
            scores = []
            steps = np.ceil(len(batch_x)/32)
            if self.tta > 1:
                for i in range(self.tta):
                    score = self.model.predict_generator(self.tta_datagen.flow(batch_x), steps)
                    scores.append(score)
            else:
                score = self.model.predict_generator(self.datagen.flow(batch_x), steps)
                scores.append(score)

            mean_score = np.mean(scores, axis=0)
            pred = np.argmax(mean_score, axis=1)

            for i in range(len(all_faces)):
                # 给score显示条形图
                # =======================================================
                if show_score:
                    plt.subplot(1, 2, 2)
                    plt.subplots_adjust(0.125, 0.1, 0.9, 0.9, 0.2, 0.2)
                    plt.bar(range(class_num), mean_score[i], align='center', color='steelblue', alpha=0.8)
                    plt.ylabel('Score')
                    plt.xticks(range(class_num), class_names)
                    # plt.show()
                # ========================================================
                for mark in all_landmarks[i]:
                    cv2.circle(img, (mark[0], mark[1]), 1, (0, 255, 0), -1, 8)
                cv2.putText(img=img, text=class_names[pred[i]],
                            org=(all_landmarks[i][21][0], all_landmarks[i][21][1] - 20),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.6,
                            color=(0, 0, 255))
        return img

    def detect_camera(self):
        cam = cv2.VideoCapture(0)
        while True:
            sucess, img = cam.read()
            if not sucess:
                continue
            img = self._detect_sentiment(img)
            if img is not None:
                cv2.imshow("Sentiment Detection", img)
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
            elif k == ord("s"):
                cv2.imwrite("capture.jpg", img)
                cv2.destroyAllWindows()
                break
        # 关闭摄像头
        cam.release()

    def detect_image(self, img_path):
        plt.figure(figsize=(12, 5))
        img = cv2.imread(img_path)
        img = self._detect_sentiment(img, show_score=True)
        plt.subplot(1, 2, 1)
        plt.subplots_adjust(0.125, 0.1, 0.9, 0.9, 0.2, 0.2)
        plt.title('Expression Detection Result')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        path = os.path.splitext(img_path)[0] + "_result.png"
        plt.savefig(path)
        plt.show()
        # cv2.imshow('Expression Detection Result', img)
        # cv2.destroyAllWindows()

    def detect(self):
        if self.cam:
            self.detect_camera()
        else:
            self.detect_image(self.img_path)


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() == 'True':
            return True
        elif v.lower() == 'False':
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser()
    parser.add_argument('-tta', '--tta', default=1, type=int, help='Test-time augmentation times. Default 1.')
    parser.add_argument('-model', '--model_name', default='vgg16', type=str,
                        help='The classification model. Default vgg16.')
    parser.add_argument('-cam', '--camera', default=False, type=str2bool,
                        help='Whether to detect face using camera. Default False')
    parser.add_argument('-path', '--image_path', default='../data/demo/sad.jpg', type=str,
                        help='The path of the image. Only useful when -cam=false.')

    args = parser.parse_args()

    test_args = dict()
    test_args['tta'] = args.tta
    test_args['cam'] = args.camera
    test_args['model'] = args.model_name
    test_args['path'] = args.image_path

    print(args)
    detector = Detector(**test_args)
    detector.detect()
