# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:04:33 2019

@author: ycai
"""

import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
os.chdir('{}/codes'.format(home))
import xgb_model
import pandas as pd

datasets = ['breast_cancer', 'diabetes', 'covtype', 'cod_rna', 'ijcnn1', 'Sensorless', 'webspam', 'MNIST', 'Fashion_MNIST', 'MNIST2_6']
n = len(datasets)
nclasses = [2, 2,  7, 2, 2, 11, 2, 10, 10, 2]
sample_size = [100, 100, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 1000]
deps = [6,5,8,4,8,6,8, 8, 8, 4]
trees = [10,20,80,80,60,30,100, 200, 200, 1000]
parameters = []
for i in range(n):
    pdict = {'max_dep':deps[i], 'ntrees':trees[i]}
    parameters.append(pdict)


test_accuracies = []
used_time = []
for i in range(n):
    model, accuracy, sample_df, time_used = xgb_model.xgb_model(datasets[i], parameters[i], nclasses[i], sample_size[i])
    used_time.append(time_used)
    test_accuracies.append(accuracy)
    os.chdir('{}/chosen_sample/xgb'.format(home))
    sample_df.to_pickle('{}_xgb_samples.pkl'.format(datasets[i]))
    os.chdir('{}/models/xgb'.format(home))
    model.save_model('{}_xgb.model'.format(datasets[i]))  
    print('{} is done'.format(datasets[i]))
    
    
accu_df = pd.DataFrame()
accu_df['datasets'] = datasets
accu_df['test accuracy'] = test_accuracies
accu_df['used time'] = used_time
os.chdir('{}/test_accu'.format(home))
accu_df.to_excel('xgb_accu.xlsx')
accu_df.to_csv('xgb_accu.txt')


