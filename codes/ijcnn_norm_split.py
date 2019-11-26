# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:28:54 2019

@author: ycai
"""

import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
import numpy as np
os.chdir('{}/data'.format(home))
import pandas as pd

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

dataset = 'ijcnn1'

file_name = 'ijcnn1_train.tr'

df = pd.read_csv(file_name, sep = ' ', header = None)
Y = df[0].tolist()
X = df.drop(columns = [0])
temp = pd.DataFrame(columns = np.arange(1, 23), index = range(X.shape[0]))
temp = temp.fillna(0.0)

for i in range(X.shape[0]):
        for j in X.columns.tolist():
            current = X.at[i, j]
            try:
                index = current.index(':')
                column = int(current[:index])
                temp.at[i,column] = current[index+1:]
            except ValueError:
                continue        

X = temp

X = X.astype('float')
X = normalize_data(X)
df = X
df['label'] = Y
    
index = file_name.index('.')
name = file_name[:index]        
try:
    os.mkdir(dataset)
except OSError:
    pass
os.chdir('{}/data/{}'.format(home, dataset))
df.to_pickle('{}_df.pkl'.format(name))


file_name = 'ijcnn1_test.t'

df = pd.read_csv(file_name, sep = ' ', header = None)
Y = df[0].tolist()
X = df.drop(columns = [0])
temp = pd.DataFrame(columns = np.arange(1, 23), index = range(X.shape[0]))
temp = temp.fillna(0.0)

for i in range(X.shape[0]):
        for j in X.columns.tolist():
            current = X.at[i, j]
            try:
                index = current.index(':')
                column = int(current[:index])
                temp.at[i,column] = current[index+1:]
            except ValueError:
                continue        

X = temp

X = X.astype('float')
X = normalize_data(X)
df = X
df['label'] = Y
    
index = file_name.index('.')
name = file_name[:index]        
try:
    os.mkdir(dataset)
except OSError:
    pass
os.chdir('{}/data/{}'.format(home, dataset))
df.to_pickle('{}_df.pkl'.format(name))
    


