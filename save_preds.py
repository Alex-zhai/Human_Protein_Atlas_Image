# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/31 11:01

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import data_feed
import model_feed

IMAGE_SIZE = 299


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

    for name in tqdm(submit['Id']):
        path = os.path.join("test", name)
        image = data_feed.load_image(path, (IMAGE_SIZE, IMAGE_SIZE, 3)) / 255.
        score_predict = tta(model, image)
        predicted.append(score_predict)

    np_save_path = "pred_results/" + save_model_path.split("/")[-1].split(".")[0] + ".npy"
    np.save(np_save_path, np.asarray(predicted))
    return np.asarray(predicted)
    # return np.load(np_save_path)


if __name__ == '__main__':
    model = model_feed.create_inception_res_v2_model()
    save_model_path = "saved_model/inception_res.h5"
    get_single_model_result(model, save_model_path)
