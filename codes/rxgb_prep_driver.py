# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:13:31 2019

@author: Ying
"""

import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
os.chdir('{}/codes'.format(home))
import rxgb_prep
import time
import subprocess
import pandas as pd

datasets = datasets = ['breast_cancer', 'diabetes', 'covtype', 'cod_rna', 'ijcnn1', 'Sensorless', 'webspam', 'MNIST', 'Fashion_MNIST', 'MNIST2_6']
n = len(datasets)
nclasses = [2, 2,  7,  2, 2, 11, 2, 10, 10, 2]
nfs = [ 9, 8, 54, 8, 22, 48, 254, 784, 784, 784]
sample_size = [100, 100, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 1000]
deps = [8,5,8,5,8,6,8, 8, 8, 6]
trees = [10,20,80,80,60,30,100, 200, 200, 1000]
eps = [0.3, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.3, 0.1, 0.3]
eta = [0.2, 0.2, 0.3, 0.1, 1.0, 0.3, 0.3, 0.3, 0.3, 0.01]
gamma = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0, 0, 0]
parameters = []
for i in range(n):
    pdict = {'max_dep':deps[i], 'ntrees':trees[i], 'eps':eps[i], 'eta':eta[i], 'gamma':gamma[i]}
    if nclasses[i] == 2:
        pdict['objective'] = 'binary:logistic'
    else:
        pdict['objective'] = 'multi:softmax'
        
    if nclasses[i] > 2:
        pdict['num_class'] = nclasses[i]
    parameters.append(pdict)

times = []
for i in range(n):
    rxgb_prep.make_conf(datasets[i], parameters[i])
    if nclasses[i] == 2:
        binary_class = True
    else:
        binary_class = False
    rxgb_prep.df2svm(datasets[i], binary_class, nfs[i])
    
    os.chdir('{}/models/rxgb'.format(home))
    try:
        os.mkdir('{}'.format(datasets[i]))
    except OSError:
        pass
    
    os.chdir('{}/models/rxgb/{}'.format(home, datasets[i]))
    subprocess.call(['cd', '{}/models/rxgb/{}'.format(home, datasets[i])])
    start = time.time()
    subprocess.call(['{}/RobustTrees/xgboost'.format(home), '{}/data/rxgb/{}.conf'.format(home, datasets[i])])
    end = time.time()
    times.append(end - start)
    output = subprocess.check_output(['ls'])
    t = str(output)
    t1 = t[t.index("'")+1:t.index("\\")]
    os.rename(t1, '{}_rxgb.model'.format(datasets[i]))
    print('{} is done'.format(datasets[i]))

t_df = pd.DataFrame()
t_df['dataset'] = datasets
t_df['used time'] = times
os.chdir('{}/test_accu'.format(home))
t_df.to_csv('rxgb_used_time.txt')
t_df.to_excel('rxgb_used_time.xlsx')