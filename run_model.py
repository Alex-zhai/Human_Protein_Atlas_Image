# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/24 16:54

import os
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from tqdm import tqdm

import data_feed
import model_feed

IMAGE_SIZE = 299


def create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), nb_classes=28):
    return model_feed.create_model(input_shape, nb_classes)


def get_train_eval_data_generator(data_path="train", data_csv_path="train.csv"):
    train_dataset_info, val_dataset_info = data_feed.get_dataset_info(data_path, data_csv_path)
    train_dataset_gen = data_feed.batch_generator(train_dataset_info, batch_size=16, shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                  augument=True)
    val_dataset_gen = data_feed.batch_generator(val_dataset_info, batch_size=32, shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                augument=False)
    return train_dataset_gen, val_dataset_gen


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def tta(model, x):
    # x is [299, 299, 3] shape
    def rotate_image(im, angle):
        if angle % 90 == 0:
            angle = angle % 360
            if angle == 0:
                return im
            elif angle == 90:
                return im.transpose((1, 0, 2))[:, ::-1, :]
            elif angle == 180:
                return im[::-1, ::-1, :]
            elif angle == 270:
                return im.transpose((1, 0, 2))[::-1, :, :]

    pred = model.predict(np.expand_dims(x, axis=0))[0]
    lr_pred = model.predict(np.expand_dims(np.fliplr(x), axis=0))[0]
    ud_pred = model.predict(np.expand_dims(np.flipud(x), axis=0))[0]
    rot90_pred = model.predict(np.expand_dims(rotate_image(x, 90), axis=0))[0]
    rot180_pred = model.predict(np.expand_dims(rotate_image(x, 180), axis=0))[0]
    rot270_pred = model.predict(np.expand_dims(rotate_image(x, 270), axis=0))[0]
    final_pred = (pred + lr_pred + ud_pred + rot90_pred + rot180_pred + rot270_pred) / 6
    return final_pred


def stage1_2_train(save_model_path):
    model = create_model()
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True
    model.layers[-5].trainable = True
    model.layers[-6].trainable = True

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    train_dataset_gen, val_dataset_gen = get_train_eval_data_generator()
    checkpoint = ModelCheckpoint(filepath=save_model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                  verbose=1, mode='auto', epsilon=0.0001)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=6)
    callbacks_list = [checkpoint, early, reduce_lr]

    model.fit_generator(train_dataset_gen, steps_per_epoch=10000, validation_data=val_dataset_gen,
                        validation_steps=1000, epochs=2, verbose=1)

    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.fit_generator(train_dataset_gen, steps_per_epoch=10000, validation_data=val_dataset_gen,
                        validation_steps=1000, epochs=10, verbose=1, callbacks=callbacks_list)


def stage_3_train(pre_model_path, save_model_path):
    model = create_model()
    train_dataset_gen, val_dataset_gen = get_train_eval_data_generator()
    checkpoint = ModelCheckpoint(filepath=save_model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=3, verbose=1, mode='auto')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=6)
    callbacks_list = [checkpoint, early, reduce_lr]
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005), metrics=['acc', f1])
    model.load_weights(pre_model_path)
    model.fit_generator(train_dataset_gen, steps_per_epoch=10000, validation_data=val_dataset_gen,
                        validation_steps=1000, epochs=10, verbose=1, callbacks=callbacks_list)

    submit = pd.read_csv("sample_submission.csv")
    predicted = []
    model.load_weights(save_model_path)

    for name in tqdm(submit['Id']):
        path = os.path.join("test", name)
        image = data_feed.load_image(path, (IMAGE_SIZE, IMAGE_SIZE, 3)) / 255.
        # score_predict = model.predict(np.expand_dims(image, axis=0))[0]
        score_predict = tta(model, image)
        label_predict = np.arange(28)[score_predict >= 0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)

    submit['Predicted'] = predicted
    submit.to_csv('submit_InceptionV3_stage3.csv', index=False)


if __name__ == '__main__':
    # stage1_2_train(save_model_path="saved_model/InceptionV3.h5")
    stage_3_train(pre_model_path="saved_model/InceptionV3.h5", save_model_path="saved_model/stage3_InceptionV3.h5")
