# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:54:32 2019

@author: ycai
"""

import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
import numpy as np
from sklearn.model_selection import train_test_split
os.chdir('{}/data'.format(home))
import pandas as pd
from sklearn.datasets import load_svmlight_file

def normalize_data(X):
    features = X.columns.tolist()
    new_df = pd.DataFrame(columns = features)
    for f in features:
        current = X[f].tolist()
        min_v = np.min(current)
        max_v = np.max(current)
        delta = max_v - min_v
        if delta != 0:
            temp = (current - min_v) / delta
        else:
            temp = current
        new_df[f] = temp
    return new_df

def split_dataset(df, test_size):
    train, test = train_test_split(df, test_size = test_size)
    return train, test

data = load_svmlight_file('webspam_wc_normalized_unigram.svm', 254)
X = data[0]
y = data[1]

X = pd.DataFrame(X.toarray())
X = X.astype('float')
X = normalize_data(X)


df = X
df['label'] = y
df['label'] = df['label'].astype('int')

train, test = split_dataset(df, 50000)
train = train.sort_index()
test = test.sort_index()

dataset = 'webspam' 
try:
    os.mkdir(dataset)
except OSError:
    pass

os.chdir('{}/data/'.format(home, dataset))
train.to_pickle('{}_train_df.pkl'.format(dataset))
test.to_pickle('{}_test_df.pkl'.format(dataset))
