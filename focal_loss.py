# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/29 20:06

import tensorflow as tf
import keras.backend as K

tf.enable_eager_execution()


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    print(pt_1)
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    print(pt_0)
    loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return loss


def focal_loss1(y_true, y_pred):
    gamma = 2.

    max_val = K.clip(-y_pred, 0, 1)
    print(max_val)
    loss = y_pred - y_pred * y_true + max_val + K.log(K.exp(-max_val) + K.exp(-y_pred - max_val))
    invprobs = tf.log_sigmoid(-y_pred * (y_true * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss

    return K.mean(K.sum(loss, axis=1))


# y = tf.constant([1, 0, 0])
# pred = tf.constant([0.9, 0.1, 0.1])

Y_true = tf.constant([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=tf.float32)
Y_pred = tf.constant([[0, 0.9, 0, 0.9, 0], [1, 0.2, 1, 0, 1]], dtype=tf.float32)

print(focal_loss(Y_true, Y_pred))
