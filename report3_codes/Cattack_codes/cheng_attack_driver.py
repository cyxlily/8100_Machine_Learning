# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:14:27 2019

@author: Ying
"""

import argparse

import pandas as pd
import xgboost as xgb
import numpy as np
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
os.chdir('{}/codes'.format(home))
import cheng_attack
from sklearn.datasets import load_svmlight_file
import scipy
import time

def predict(model, sampleX, nclasses):
    dtest = xgb.DMatrix(scipy.sparse.csr_matrix(sampleX))
    prediction = model.predict(dtest)
    if nclasses == 2:
        if prediction<0.5:
            prediction = 0
        else:
            prediction = 1
    else:
        prediction = int(prediction)
    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name', type=str, help = 'name of the data set')
    parser.add_argument('testset', type = str, help='test data path')
    parser.add_argument('m', type = str, help='model path')
    parser.add_argument('c', type=int, help='number of classes')
    parser.add_argument('n', type=int, help='number of features')
    parser.add_argument('--feature_start', type=int, default=0, choices=[0,1], help='feature number starts from which index? For cod-rna and higgs, this should be 0.')

    args = parser.parse_args()
    d_name = args.data_name
    data_path = args.testset
    model_path = args.m
    nclasses = args.c
    nfeatures = args.n
    f_start = args.feature_start

    model = xgb.Booster()
    model.load_model(model_path)
    
    test_data, test_label = load_svmlight_file(data_path, n_features = nfeatures)
    test_data = test_data.toarray()
    test_label = test_label.astype('int')
    n = len(test_label)
    df = pd.DataFrame(test_data)
    df['label'] = test_label
    df = df.sample(frac=1)
    test_label = df['label'].tolist()
    test_data = np.array(df.drop(columns=['label']))
    
    if n<200:
        n_selected = n
    else:
        n_selected = 200
        
        
    ori_points = []
    results = []
    used_time = []
    count = 0
    tt = 0
    while count<n_selected:        
        s = test_data[tt]
        sl = test_label[tt]
        if predict(model, s, nclasses) == sl:        
            start = time.time()
            r = cheng_attack.attack(model, test_data, test_label, s, sl, nclasses, tt)
            end = time.time()
            used_time.append(end - start)
            ori_points.append(s)
            results.append(r)
            count += 1
            print('sample {} is done'.format(tt))
        tt += 1

    total_dis = 0
    pert = pd.DataFrame()
    index = []
    points = []
    dis = []
    for (j, d, p) in results:
        index.append(j)
        points.append(p)
        dis.append(d)
        total_dis += d


    pert['index'] = index
    pert['distance'] = dis
    pert['used_time'] = used_time
    pert['pert point'] = points
    pert['ori point'] = ori_points
    os.chdir('{}/attack/cheng'.format(home))
    pert.to_csv('{}_cheng_attack_xgb.txt'.format(d_name))
    with open('{}_cheng_xgb_ave.txt'.format(d_name), 'w') as f:
        f.write('average distance: ' + str(total_dis/len(test_label)))
        f.write('\n\naverage used time: ' + str(np.mean(pert['used_time'])))

    f.close()
    print('{} is done'.format(d_name))