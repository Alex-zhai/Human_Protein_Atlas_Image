# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/24 16:55

from __future__ import print_function, absolute_import, division

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_image(path, shape=(299, 299, 3)):
    image_red_ch = Image.open(path + '_red.png')
    image_green_ch = Image.open(path + '_green.png')
    image_blue_ch = Image.open(path + '_blue.png')
    image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch)), -1)
    # image = Image.fromarray(image)
    image = cv2.resize(image, (shape[0], shape[1]))
    # image = image.resize((shape[0], shape[1]), resample=Image.BICUBIC)
    return image


def load_4channel_image(path, shape=(299, 299, 4)):
    image_red_ch = Image.open(path + '_red.png')
    image_green_ch = Image.open(path + '_green.png')
    image_blue_ch = Image.open(path + '_blue.png')
    image_yellow_ch = Image.open(path + '_yellow.png')
    image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch),
        np.array(image_yellow_ch)), -1)
    # image = Image.fromarray(image)
    image = cv2.resize(image, (shape[0], shape[1]))
    # image = image.resize((shape[0], shape[1]), resample=Image.BICUBIC)
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


def get_dataset_info(path_to_train_data, path_to_train_csv):
    dataset_info = []
    data = pd.read_csv(path_to_train_csv)
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        dataset_info.append({
            'path': os.path.join(path_to_train_data, name),
            'labels': np.array([int(label) for label in labels])})
    dataset_info = np.array(dataset_info)

    # split train and val
    indexes = np.arange(dataset_info.shape[0])
    np.random.shuffle(indexes)
    train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=42)
    train_dataset_info = dataset_info[train_indexes]
    val_dataset_info = dataset_info[valid_indexes]

    return train_dataset_info, val_dataset_info


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


def rgby_batch_generator(dataset_info, batch_size, shape, augument=True):
    assert shape[2] == 4
    while True:
        dataset_info = shuffle(dataset_info)
        for start in range(0, len(dataset_info), batch_size):
            end = min(start + batch_size, len(dataset_info))
            batch_images = []
            x_batch = dataset_info[start:end]
            batch_labels = np.zeros((len(x_batch), 28))
            for i in range(len(x_batch)):
                image = load_4channel_image(x_batch[i]['path'], shape)
                if augument:
                    image = augment(image)
                batch_images.append(image / 255.0)
                batch_labels[i][x_batch[i]['labels']] = 1
            yield np.array(batch_images, np.float32), batch_labels
