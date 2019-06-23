# coding=utf-8
import config
import sklearn.metrics as metrics
import keras
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import pandas as pd
import datetime
import time
import os
from decimal import Decimal
from tqdm import tqdm
from models import *
import argparse


def train(**kwargs):
    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)

    c = K.tf.ConfigProto()
    c.gpu_options.allow_growth = True
    session = K.tf.Session(config=c)
    set_session(session)

    train_df = pd.read_pickle(config.TRAIN_PATH)
    valid_df = pd.read_pickle(config.VALID_PATH)

    train_size = len(train_df)
    valid_size = len(valid_df)

    train_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        brightness_range=(0.8, 1.2),
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    valid_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
    )

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=config.TRAIN_DIR,
        x_col='path',
        y_col='emotion',
        target_size=(config.img_size, config.img_size),
        color_mode='grayscale',
        classes=list(range(config.class_num)),
        batch_size=kwargs['batch_size']
    )

    valid_gen = valid_datagen.flow_from_dataframe(
        valid_df,
        directory=config.VALID_DIR,
        x_col='path',
        y_col='emotion',
        target_size=(config.img_size, config.img_size),
        color_mode='grayscale',
        classes=list(range(config.class_num)),
        batch_size=kwargs['batch_size']
    )

    model_save_path = config.MODEL_DIR + '/' + 'model_{}.h5'.format(kwargs['model'])

    # Set all the callbacks.
    Fname = 'Face_'
    Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    tensorboard = TensorBoard(log_dir=config.LOG_DIR + '/' + Time,
                              histogram_freq=0, write_graph=False, write_images=False,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='min', baseline=None,
                        restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4)

    model = model_func[kwargs['model']](kwargs['dropout'], kwargs['lr'])
    print(model.summary())

    steps_per_epoch = np.ceil(train_size / kwargs['batch_size'])
    validation_steps = np.ceil(valid_size / kwargs['batch_size'])
    model.fit_generator(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=300,
                        callbacks=[ear, checkpoint, tensorboard, reduce_lr],
                        validation_data=valid_gen,
                        validation_steps=validation_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dropout', '--dropout_rate', default=0.5, type=float,
                        help='The dropout rate for the last dense layers.'
                             'Default 0.5.')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate. Default 1e-3.')
    parser.add_argument('-batch_size', '--batch_size', default=128, type=int, help='Batch size. Default 128.')
    parser.add_argument('-model', '--model_name', default='vgg16', type=str,
                        help='The classification model. Default vgg16.')
    model_func = {'vgg16': setVGG16, 'xception': setXception, 'resnet50': setResNet50}
    args = parser.parse_args()

    train_args = dict()
    train_args['dropout'] = args.dropout_rate
    train_args['lr'] = args.learning_rate
    train_args['batch_size'] = args.batch_size
    train_args['model'] = args.model_name

    print(args)
    train(**train_args)






