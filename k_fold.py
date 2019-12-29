# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/11/2 13:35

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from itertools import chain
from collections import Counter

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
train_df = pd.read_csv("train.csv")
print(train_df.index.values)
print(train_df.head(1))


def show_df(indices):
    train_df = pd.read_csv("train.csv")
    x_df = train_df.loc[indices]

    x_df['path'] = x_df['Id'].map(lambda x: os.path.join("train", '{}.rgb'.format(x)))
    x_df['target_list'] = x_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
    all_labels = list(chain.from_iterable(x_df['target_list'].values))
    c_val = Counter(all_labels)
    print(c_val)
    n_keys = c_val.keys()
    max_idx = max(n_keys)
    x_df['target_vec'] = x_df['target_list'].map(lambda ck: [i in ck for i in range(max_idx + 1)])


for i, (x, y) in enumerate(k_fold.split(train_df.index.values, train_df.Target)):
    print(i)
    show_df(x)
    print("dadsad")
    show_df(y)
