# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/29 20:53

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import data_feed
import model_feed

IMAGE_SIZE = 299


def get_ensemble_models(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), nb_classes=28):
    # now we have inception and resnet50 model
    models = [model_feed.create_model(input_shape, nb_classes),
              model_feed.create_resnet50_model(input_shape, nb_classes)]
    paths = ["saved_model/InceptionV3.h5", "saved_model/resnet50.h5"]
    return models, paths


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


def get_single_model_result(model, save_model_path):
    submit = pd.read_csv("sample_submission.csv")
    predicted = []
    model.load_weights(save_model_path)

    for name in tqdm(submit['Id'][:10]):
        path = os.path.join("test", name)
        image = data_feed.load_image(path, (IMAGE_SIZE, IMAGE_SIZE, 3)) / 255.
        score_predict = tta(model, image)
        predicted.append(score_predict)
    return np.asarray(predicted)


def submit_result():
    submit = pd.read_csv("sample_submission.csv")
    models, paths = get_ensemble_models()
    model_num = len(models)
    sum_predicted = 0.0
    for model, path in zip(models, paths):
        sum_predicted += get_single_model_result(model, path)
    aver_predicted = sum_predicted / model_num
    final_predicted = []
    for pred in aver_predicted.tolist():
        label_predict = np.arange(28)[pred >= 0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        final_predicted.append(str_predict_label)

    submit['Predicted'] = final_predicted
    submit.to_csv('submit_InceptionV3_more_TTA_model_ensemble.csv', index=False)


if __name__ == '__main__':
    submit_result()
