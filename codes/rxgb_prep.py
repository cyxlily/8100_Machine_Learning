# -*- coding: utf-8 -*-
"""
Created on Sun Oct 8 10:05:46 2019

@author: Ying
"""


import numpy as np
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
import pandas as pd
from sklearn.datasets import dump_svmlight_file

def make_conf(dataset, parameters):
    text = 'booster = gbtree \ntree_method = robust_exact\nrobust_training_verbose = false\nsave_period=0\n'
    if 'num_class' in parameters.keys():
        text = text + 'num_class = {}\n'.format(parameters['num_class'])
    text = text + 'objective = {}\n'.format(parameters['objective'])
    text = text + 'robust_eps = {}\neta = {}\ngamma = {}\n'.format(parameters['eps'], parameters['eta'], parameters['gamma'])
    text = text + 'max_depth = {}\n'.format(parameters['max_dep'])
    text = text + 'num_round = {}\n'.format(parameters['ntrees'])    
    text = text + 'data = "/home/cai7/data/rxgb/{}_train.svm"\n'.format(dataset)
    text = text + 'test:data = "/home/cai7/data/rxgb/{}_test.svm"\n'.format(dataset)
    text = text + 'eval[test] = "/home/cai7/data/rxgb/{}_test.svm"\n'.format(dataset)
    with open('{}/data/rxgb/{}.conf'.format(home, dataset), 'w+') as f:
        f.write(text)
    
    f.close()

def df2svm(dataset, binary_class, nfs):
    os.chdir('{}/data/{}'.format(home, dataset))
    train = pd.read_pickle('{}_train_df.pkl'.format(dataset))
    test = pd.read_pickle('{}_test_df.pkl'.format(dataset))
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)    
    if binary_class == True:
        #print('binary class')
        classes = np.unique(train['label'].tolist())
        if len(classes) != 2:
            print('Error: binary classes n != 2')
            exit()
        if ((1 not in classes) or (0 not in classes)):
            class0 = classes[0]            
            y = train['label'].tolist()
            for i in range(len(y)):
                if y[i] == class0:
                    y[i] = 0
                else:
                    y[i] = 1
            train['label'] = y
            y = test['label'].tolist()
            for i in range(len(y)):
                if y[i] == class0:
                    y[i] = 0
                else:
                    y[i] = 1
            test['label'] = y
    if binary_class == False:
        #print('Multiple classes')
        classes = np.unique(train['label'].tolist())
        if not isinstance(classes[0], str):
            minclass = np.min(classes)
            if minclass != 0 :
                y = train['label'].tolist()
                for i in range(len(y)):
                    y[i] = y[i] - minclass
                train['label'] = y
                y = test['label'].tolist()
                for i in range(len(y)):
                    y[i] = y[i] - minclass
                test['label'] = y
    
    zbased = False
    if ('HIGGS' in dataset or 'rna' in dataset):
        zbased = True
    
      
    X = train.drop(columns = ['label'])
    y = train['label']
    if ('cov' in dataset or 'MNIST' in dataset or 'webspam' in dataset):     
        t = X.iloc[0].tolist()
        b = [v + 0.000000001 for v in t]
        X.iloc[0] = b    
    X = np.array(X)
    y = np.array(y)
    dump_svmlight_file(X, y, '{}/data/rxgb/{}_train.svm'.format(home, dataset), zero_based = zbased)    
    X = test.drop(columns = ['label'])
    y = test['label']
    if ('cov' in dataset or 'MNIST' in dataset or 'webspam' in dataset):     
        t = X.iloc[0].tolist()
        b = [v + 0.000000001 for v in t]
        X.iloc[0] = b
    X = np.array(X)
    y = np.array(y)
    dump_svmlight_file(X, y, '{}/data/rxgb/{}_test.svm'.format(home, dataset), zero_based = zbased)


