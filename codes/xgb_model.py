# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:59:06 2019

@author: ycai
"""

import xgboost
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]

import pandas as pd
import random
import numpy as np
import time


def train_model(train_df, parameters, nclasses, test_df):
    Y = train_df['label'].tolist()
    X = train_df.drop(columns = ['label'])
    dtrain = xgboost.DMatrix(X, label = Y)    
    Y = test_df['label'].tolist()
    X = test_df.drop(columns = ['label'])
    dtest = xgboost.DMatrix(X, label = Y)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    max_dep = parameters['max_dep']
    num_round = parameters['ntrees']
    param = {'max_depth': max_dep}    
    if nclasses == 2:
        param['objective'] = 'binary:logistic'
        model = xgboost.train(param, dtrain, num_round, evallist)
    else:
        param['objective'] = 'multi:softmax'
        param['num_class'] = nclasses
        model = xgboost.train(param, dtrain, num_round, evallist)    
    return model
    
def test_performance(model, test_df, nclasses):
    truey = test_df['label']
    X = test_df.drop(columns = ['label'])
    dtest = xgboost.DMatrix(X)    
    prediction = model.predict(dtest)
    if nclasses == 2:
        temp = []
        for v in prediction:
            if v<0.5:
                temp.append(0)
            else:
                temp.append(1)
        prediction = temp
    else:
        prediction = prediction.astype('int')
    temp = pd.DataFrame()
    temp['true'] = truey
    temp['pred'] = prediction
    correct = 0
    correct_classified = []
    for i in range(temp.shape[0]):
        if temp.iloc[i]['true'] == temp.iloc[i]['pred']:
            correct = correct + 1
            correct_classified.append(i)
    return (correct / temp.shape[0]), correct_classified, prediction
    
def select_samples(correct_classified, sample_size):
    temp = random.sample(correct_classified, sample_size)    
    return temp
    
def xgb_model(dataset, parameters, nclasses, sample_size):
    os.chdir('{}/data/'.format(home, dataset))
    train_df = pd.read_pickle('{}_train_df.pkl'.format(dataset))
    test_df = pd.read_pickle('{}_test_df.pkl'.format(dataset))
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    classes = np.unique(train_df['label'].tolist())
    if ((nclasses == 2) and ((1 not in classes) or (0 not in classes))):
        class0 = classes[0]            
        y = train_df['label'].tolist()
        for i in range(len(y)):
            if y[i] == class0:
                y[i] = 0
            else:
                y[i] = 1
        train_df['label'] = y
        y = test_df['label'].tolist()
        for i in range(len(y)):
            if y[i] == class0:
                y[i] = 0
            else:
                y[i] = 1
        test_df['label'] = y
    if (nclasses > 2):
        min_c = np.min(classes)
        if min_c > 0:
            y = train_df['label'].tolist()
            temp = []
            for i in range(len(y)):
                temp.append(int(y[i] - min_c))
            train_df['label'] = temp
            y = test_df['label'].tolist()
            temp = []
            for i in range(len(y)):
                temp.append(int(y[i] - min_c))
            test_df['label'] = temp
    start = time.time()
    model = train_model(train_df, parameters, nclasses, test_df)
    end = time.time()    
    accuracy, correct_classified, pred = test_performance(model, test_df, nclasses)
    samples = select_samples(correct_classified, sample_size)
    sample_df = test_df.iloc[samples, :]
    sample_df = sample_df.sort_index()
    
    return model, accuracy, sample_df, (end - start)
    
    
    
    
    