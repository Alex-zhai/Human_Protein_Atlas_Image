# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/24 17:26

from __future__ import print_function, absolute_import, division

from keras import layers, models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.xception import Xception


def create_model(input_shape=(299, 299, 3), nb_classes=28):
    input_img = layers.Input(shape=input_shape)
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    x = layers.BatchNormalization()(input_img)
    x = base_model(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(nb_classes, activation='sigmoid')(x)
    return models.Model(input_img, out)


def create_vgg19_model(input_shape=(299, 299, 3), nb_classes=28):
    input_img = layers.Input(shape=input_shape)
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    x = layers.BatchNormalization()(input_img)
    x = base_model(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(nb_classes, activation='sigmoid')(x)
    return models.Model(input_img, out)


def create_inception_res_v2_model(input_shape=(299, 299, 3), nb_classes=28):
    input_img = layers.Input(shape=input_shape)
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    x = layers.BatchNormalization()(input_img)
    x = base_model(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(nb_classes, activation='sigmoid')(x)
    return models.Model(input_img, out)


def create_densenet_model(input_shape=(299, 299, 3), nb_classes=28, dense_num=121):
    input_img = layers.Input(shape=input_shape)
    if dense_num == 121:
        base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    elif dense_num == 169:
        base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape)
    x = layers.BatchNormalization()(input_img)
    x = base_model(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(nb_classes, activation='sigmoid')(x)
    return models.Model(input_img, out)


def create_xception_model(input_shape=(299, 299, 3), nb_classes=28):
    input_img = layers.Input(shape=input_shape)
    base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = layers.BatchNormalization()(input_img)
    x = base_model(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(nb_classes, activation='sigmoid')(x)
    return models.Model(input_img, out)


if __name__ == '__main__':
    model = create_model()
    print(model.summary())
