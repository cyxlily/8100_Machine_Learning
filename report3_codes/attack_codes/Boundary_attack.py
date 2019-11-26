# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 23:00:30 2019

@author: Ying
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:51:21 2019

@author: Ying
"""

import pandas as pd
import xgboost as xgb
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
from sklearn.datasets import load_svmlight_file
import time
import argparse
import numpy as np

from art.attacks import BoundaryAttack
from art.classifiers import XGBoostClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', type=str, help = 'name of the data set')
    parser.add_argument('model_name', type=str, choices=['xgb','rxgb'], help = 'natural or robust model')
    parser.add_argument('testset', type = str, help='test data path')
    parser.add_argument('m', type = str, help='model path')
    parser.add_argument('c', type=int, help='number of classes')
    parser.add_argument('n', type=int, help='number of features')
    parser.add_argument('--feature_start', type=int, default=0, choices=[0,1], help='feature number starts from which index? For cod-rna and higgs, this should be 0.')

    args = parser.parse_args()
    d_name = args.data_name
    m_name = args.model_name
    data_path = args.testset
    model_path = args.m
    nclasses = args.c
    nfeatures = args.n
    f_start = args.feature_start
    
    model = xgb.Booster()
    model.load_model(model_path)
    
    classifier = XGBoostClassifier(model=model, clip_values=(0, 1), nb_features=nfeatures, nb_classes=nclasses)
    
    test_data, test_label = load_svmlight_file(data_path, n_features = nfeatures)
    test_data = test_data.toarray()
    test_label = test_label.astype('int')
    n = len(test_label)
    df = pd.DataFrame(test_data)
    df['label'] = test_label
    df = df.sample(frac=1)
    test_label = df['label'].tolist()
    test_data = np.array(df.drop(columns=['label']))   
    
    predictions = np.argmax(classifier.predict(test_data), axis=1)
    attack = BoundaryAttack(classifier=classifier, targeted=False, delta=0.05, epsilon=0.05, step_adapt=0.5)
    n_selected = 100
    corrected = []
    c_labels = []
    for i in range(len(test_label)):
        if test_label[i] == predictions[i]:
            corrected.append(test_data[i])
            c_labels.append(test_label[i])
        if len(corrected) >= n_selected:
            break
    corrected = np.array(corrected)
    start = time.time()
    test_adv = attack.generate(corrected)
    end = time.time()
    
    avg_time = (end-start) / test_adv.shape[0]
    
    dis = 0
    for i in range(test_adv.shape[0]):
        d = np.max(np.abs(corrected[i] - test_adv[i]))
        print(str(d))
        dis += d
    avg_dis = dis / test_adv.shape[0]
    
    os.chdir('{}/attack/Boundary'.format(home))
    with open('{}_Boundary_{}_ave.txt'.format(d_name, m_name), 'w') as f:
        f.write('average distance: ' + str(avg_dis))
        f.write('\n\naverage used time: ' + str(avg_time))
        
    f.close()
    print('{} is done'.format(d_name))
    
    
    