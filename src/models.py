from keras.applications import VGG16
from keras.applications import Xception
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.layers import *
import keras
import keras.layers as layers
import config
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf


def setVGG16(dropout_rate, lr):
    main_input = layers.Input([config.img_size, config.img_size, 1])

    x = layers.BatchNormalization()(main_input)
    x = layers.GaussianNoise(0.01)(x)

    base_model = VGG16(weights=None, input_tensor=x, include_top=False)

    # flatten = layers.GlobalAveragePooling2D()(base_model.output)
    flatten = Flatten()(base_model.output)

    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(flatten)
    fc = Dropout(dropout_rate)(fc)
    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(fc)
    fc = Dropout(dropout_rate)(fc)

    predictions = Dense(config.class_num, activation="softmax")(fc)

    model = keras.Model(inputs=main_input, outputs=predictions, name='vgg16')

    optimizer = keras.optimizers.Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


def setLandmarkCNN(dropout_rate, lr):
    main_input = layers.Input([68, 2])
    x = Lambda(lambda x: K.expand_dims(x))(main_input)

    x = BatchNormalization()(x)

    x = GaussianNoise(0.01)(x)

    x1 = Conv2D(4, (6, 2), activation='relu', padding='same')(x)
    x1 = Conv2D(4, (6, 2), activation='relu', padding='same')(x1)
    # x1 = MaxPool2D(padding='same')(x1)

    x2 = Conv2D(4, (4, 2), activation='relu', padding='same')(x)
    x2 = Conv2D(4, (4, 2), activation='relu', padding='same')(x2)
    # x2 = MaxPool2D(padding='same')(x2)

    con = Concatenate()([x1, x2])
    flatten = Flatten()(con)
    x = Dense(48,
              activation='relu',
              )(flatten)
    x = Dropout(dropout_rate)(x)

    predictions = Dense(config.class_num, activation="softmax")(x)

    model = keras.Model(inputs=main_input, outputs=predictions, name='landmark')

    optimizer = keras.optimizers.Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


def setXception(dropout_rate, lr):
    main_input = layers.Input([config.img_size, config.img_size, 1])

    x = layers.BatchNormalization()(main_input)

    base_model = Xception(weights=None, input_tensor=x, include_top=False)

    flatten = layers.GlobalAveragePooling2D()(base_model.output)
    # flatten = layers.Flatten()(base_model.output)

    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(flatten)
    fc = Dropout(dropout_rate)(fc)
    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(fc)
    fc = Dropout(dropout_rate)(fc)

    predictions = Dense(config.class_num, activation="softmax")(fc)

    model = keras.Model(inputs=main_input, outputs=predictions, name='xception')

    optimizer = keras.optimizers.Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


def setResNet50(dropout_rate, lr):
    main_input = layers.Input([config.img_size, config.img_size, 1])

    x = layers.BatchNormalization()(main_input)
    x = GaussianNoise(0.01)(x)

    base_model = ResNet50(weights=None, input_tensor=x, include_top=False)

    flatten = layers.GlobalAveragePooling2D()(base_model.output)
    # flatten = layers.Flatten()(base_model.output)

    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(flatten)
    fc = Dropout(dropout_rate)(fc)
    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(fc)
    fc = Dropout(dropout_rate)(fc)

    predictions = Dense(config.class_num, activation="softmax")(fc)

    model = keras.Model(inputs=main_input, outputs=predictions, name='resnet50')

    optimizer = keras.optimizers.Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


