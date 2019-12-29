# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/10/30 17:22

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter
from sklearn.model_selection import train_test_split

image_df = pd.read_csv('train.csv')
# print(image_df.head())
print(image_df['Id'].value_counts().shape[0])
image_df['path'] = image_df['Id'].map(lambda x: os.path.join("train", '{}.rgb'.format(x)))
image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
# print(image_df.head())

all_labels = list(chain.from_iterable(image_df['target_list'].values))
# print(all_labels)
c_val = Counter(all_labels)
print(c_val)
n_keys = c_val.keys()
# print(n_keys)
max_idx = max(n_keys)
# print(max_idx)

image_df['target_vec'] = image_df['target_list'].map(lambda ck: [i in ck for i in range(max_idx + 1)])
print(image_df.sample(3))

raw_train_df, valid_df = train_test_split(image_df,
                                          test_size=0.15,
                                          # hack to make stratification work
                                          stratify=image_df['Target'].map(lambda x: x[:3] if '27' not in x else '0'))
print(raw_train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

out_df_list = []
for k, v in c_val.items():
    if v > 100:
        keep_rows = raw_train_df['target_list'].map(lambda x: k in x)
        out_df_list += [raw_train_df[keep_rows].sample(1000, replace=True)]
    else:
        rare_rows = raw_train_df['target_list'].map(lambda x: k in x)
        out_df_list += [raw_train_df[rare_rows].sample(100, replace=True)]

train_df = pd.concat(out_df_list, ignore_index=True)
print(train_df.shape[0])
print(train_df.head(3))

new_all_labels = list(chain.from_iterable(valid_df['target_list'].values))
print(new_all_labels)
c_val = Counter(new_all_labels)
print(c_val)
print(c_val[22])
n_keys = c_val.keys()
print(n_keys)
max_idx = max(n_keys)
print(max_idx)
