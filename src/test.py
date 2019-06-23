# coding=utf-8
import config
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import keras
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import pandas as pd
import datetime
import time
import os
from tqdm import tqdm
from utils import *
import argparse
from models import *
from keras.preprocessing.image import load_img, img_to_array


def test(**kwargs):
    c = K.tf.ConfigProto()
    c.gpu_options.allow_growth = True
    session = K.tf.Session(config=c)
    set_session(session)

    test_df = pd.read_pickle(config.TEST_PATH)
    y_true = np.asarray(test_df['emotion'])
    test_size = len(test_df)

    # 读取测试集文件
    x_test = []
    for path in test_df['path']:
        x_test.append(img_to_array(load_img(config.TEST_DIR + '/' + path, color_mode='grayscale')))

    x_test = np.asarray(x_test)

    test_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        brightness_range=(0.8, 1.2),
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
    )

    model_save_path = config.MODEL_DIR + '/' + 'model_{}.h5'.format(kwargs['model'])

    model_image = keras.models.load_model(model_save_path)
    print(model_image.summary())

    steps = np.ceil(test_size / kwargs['batch_size'])
    scores = []
    for i in tqdm(range(kwargs['tta'])):
        score = model_image.predict_generator(
           test_datagen.flow(x_test, batch_size=kwargs['batch_size'], shuffle=False), steps=steps)
        scores.append(score)

    mean_score = np.mean(scores, axis=0)
    assert mean_score.shape[0] == y_true.shape[0]

    y_pred = np.argmax(mean_score, axis=-1)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    acc = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average='weighted')

    draw_confusion_matrix(y_true, y_pred)

    print('f1 score: {}, acc: {}, recall: {}'.format(f1, acc, recall))


def draw_confusion_matrix(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
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
           xticklabels=config.class_names, yticklabels=config.class_names,
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
    plt.savefig('../data/cm.jpg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tta', '--tta', default=5, type=int, help='Test-time augmentation times. Default 5.')
    parser.add_argument('-batch_size', '--batch_size', default=128, type=int, help='Batch size. Default 128.')
    parser.add_argument('-model', '--model_name', default='vgg16', type=str,
                        help='The classification model. Default vgg16.')

    args = parser.parse_args()

    test_args = dict()
    test_args['tta'] = args.tta
    test_args['batch_size'] = args.batch_size
    test_args['model'] = args.model_name

    print(args)
    test(**test_args)


