# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/30 19:10

from __future__ import print_function, absolute_import, division

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from itertools import chain
from collections import Counter


def load_image(path, shape=(299, 299, 3)):
    image_red_ch = Image.open(path + '_red.png')
    image_green_ch = Image.open(path + '_green.png')
    image_blue_ch = Image.open(path + '_blue.png')
    image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch)), -1)
    image = cv2.resize(image, (shape[0], shape[1]))
    return image


def new_load_image(path, shape=(299, 299, 3)):
    image_red_ch = Image.open(path + '_red.png')
    image_green_ch = Image.open(path + '_green.png')
    image_blue_ch = Image.open(path + '_blue.png')
    image_yellow_ch = Image.open(path + '_yellow.png')
    image = np.stack((
        np.array(image_red_ch) / 2 + np.array(image_yellow_ch) / 2,
        np.array(image_green_ch) / 2 + np.array(image_yellow_ch) / 2,
        np.array(image_blue_ch)), -1)
    image = cv2.resize(image, (shape[0], shape[1]))
    return image


def augment(image):
    augment_img = iaa.Sequential([
        iaa.OneOf([
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])], random_order=True)

    image_aug = augment_img.augment_image(image)
    return image_aug


def get_train_and_val_df(path_to_train_csv):
    image_df = pd.read_csv(path_to_train_csv)
    image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
    all_labels = list(chain.from_iterable(image_df['target_list'].values))
    c_val = Counter(all_labels)
    raw_train_df, valid_df = train_test_split(image_df, test_size=0.15,
                                              stratify=image_df['Target'].map(
                                                  lambda x: x[:3] if '27' not in x else '0'))
    out_df_list = []
    for k, v in c_val.items():
        if v > 100:
            keep_rows = raw_train_df['target_list'].map(lambda x: k in x)
            out_df_list += [raw_train_df[keep_rows].sample(1000, replace=True)]
        else:
            rare_rows = raw_train_df['target_list'].map(lambda x: k in x)
            out_df_list += [raw_train_df[rare_rows].sample(100, replace=True)]

    train_df = pd.concat(out_df_list, ignore_index=True)
    return train_df, valid_df


def get_dataset_info(path_to_train_data, path_to_train_csv, train_indices, val_indices):
    train_dataset_info = []
    val_dataset_info = []

    df = pd.read_csv(path_to_train_csv)
    train_df, val_df = df[train_indices], df[val_indices]

    for name, labels in zip(train_df['Id'], train_df['Target'].str.split(' ')):
        train_dataset_info.append({
            'path': os.path.join(path_to_train_data, name),
            'labels': np.array([int(label) for label in labels])})

    for name, labels in zip(val_df['Id'], val_df['Target'].str.split(' ')):
        val_dataset_info.append({
            'path': os.path.join(path_to_train_data, name),
            'labels': np.array([int(label) for label in labels])})

    return np.asarray(train_dataset_info), np.asarray(val_dataset_info)


def batch_generator(dataset_info, batch_size, shape, augument=True):
    assert shape[2] == 3
    while True:
        dataset_info = shuffle(dataset_info)
        for start in range(0, len(dataset_info), batch_size):
            end = min(start + batch_size, len(dataset_info))
            batch_images = []
            x_batch = dataset_info[start:end]
            batch_labels = np.zeros((len(x_batch), 28))
            for i in range(len(x_batch)):
                image = load_image(x_batch[i]['path'], shape)
                if augument:
                    image = augment(image)
                batch_images.append(image / 255.0)
                batch_labels[i][x_batch[i]['labels']] = 1
            yield np.array(batch_images, np.float32), batch_labels


if __name__ == '__main__':
    train_df, val_df = get_train_and_val_df("train.csv")
    info1, info2 = get_dataset_info("train", "train.csv")
    print(info2)
