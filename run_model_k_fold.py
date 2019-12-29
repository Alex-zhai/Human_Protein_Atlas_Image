# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/24 16:54

import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import model_feed
import new_data_feed

IMAGE_SIZE = 299


def create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), nb_classes=28):
    return model_feed.create_model(input_shape, nb_classes)


def get_train_eval_data_generator(train_indices, val_indices, data_path="train", data_csv_path="train.csv"):
    train_dataset_info, val_dataset_info = new_data_feed.get_dataset_info(data_path, data_csv_path, train_indices,
                                                                          val_indices)
    train_dataset_gen = new_data_feed.batch_generator(train_dataset_info, batch_size=16,
                                                      shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                      augument=True)
    val_dataset_gen = new_data_feed.batch_generator(val_dataset_info, batch_size=32, shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                    augument=False)
    return train_dataset_gen, val_dataset_gen


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
    num_folds = 10
    k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1234)
    train_df = pd.read_csv("train.csv")
    for fold_num, (train_index, valid_index) in enumerate(k_fold.split(train_df.index.values, train_df.Target)):

        for layer in model.layers:
            layer.trainable = False
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True
        model.layers[-3].trainable = True
        model.layers[-4].trainable = True
        model.layers[-5].trainable = True
        model.layers[-6].trainable = True

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        train_dataset_gen, val_dataset_gen = get_train_eval_data_generator(train_index, valid_index)
        checkpoint = ModelCheckpoint(filepath=save_model_path + "_" + str(fold_num + 1) + ".h5", monitor='val_loss',
                                     verbose=1, save_best_only=True,
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


def get_model_res(model, submit_df, save_model_path, fold_num):
    model_path = save_model_path + "_" + str(fold_num + 1) + ".h5"
    model.load_weights(model_path)
    pred_res = []
    for name in tqdm(submit_df['Id']):
        path = os.path.join("test", name)
        image = new_data_feed.load_image(path, (IMAGE_SIZE, IMAGE_SIZE, 3)) / 255.
        score_predict = tta(model, image)
        pred_res.append(score_predict)
    return pred_res


def submit_result(save_model_path):
    model = create_model()
    submit = pd.read_csv("sample_submission.csv")
    predicted = 0.0
    for i in range(5):
        predicted += get_model_res(model, submit, save_model_path, i)
    predicted /= 5
    final_predicted = []
    for i in range(predicted.shape[0]):
        label_predict = np.arange(28)[predicted[i] >= 0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        final_predicted.append(str_predict_label)

    submit['Predicted'] = final_predicted
    submit.to_csv('submit_InceptionV3_more_TTA_kfold5.csv', index=False)


if __name__ == '__main__':
    stage1_2_train(save_model_path="saved_model/InceptionV3_kfold5")
    submit_result(save_model_path="saved_model/InceptionV3_kfold5")
