# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:53:37 2019

@author: ycai
"""

import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
os.chdir('{}/codes'.format(home))
import xgb_model
import pickle
import pandas as pd

def test_performance(model, test_df):
    truey = test_df['label']
    X = test_df.drop(columns = ['label'])
    prediction = model.predict(X)
    temp = pd.DataFrame()
    temp['true'] = truey
    temp['pred'] = prediction
    correct = 0    
    for i in range(temp.shape[0]):
        if temp.iloc[i]['true'] == temp.iloc[i]['pred']:
            correct = correct + 1            
    return (correct / temp.shape[0])

datasets = ['breast_cancer', 'diabetes', 'covtype', 'cod_rna', 'HIGGS', 'ijcnn1', 'Sensorless', 'webspam']
n = len(datasets)
test_accuracies = []
for i in range(n):
    os.chdir('{}/models/xgb'.format(home))
    model = pickle.load(open("{}_xgb_model.pkl".format(datasets[i]), "rb"))
    os.chdir('{}/data/{}'.format(home, datasets[i]))
    test_df = pd.read_pickle('{}_test_df.pkl'.format(datasets[i]))
    accuracy = test_performance(model, test_df)
    test_accuracies.append(accuracy)
    print('{} is done'.format(datasets[i]))
    
    
accu_df = pd.DataFrame()
accu_df['datasets'] = datasets
accu_df['test accuracy'] = test_accuracies
os.chdir('/home/cai7/test_accu')
accu_df.to_excel('xgb_accu.xlsx')
accu_df.to_csv('xgb_accu.txt')