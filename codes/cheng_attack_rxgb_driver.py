# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:14:27 2019

@author: Ying
"""

import pandas as pd
import xgboost as xgb
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
os.chdir('{}/codes'.format(home))
import cheng_attack_rxgb
from sklearn.datasets import load_svmlight_file

datasets = ['breast_cancer', 'diabets', 'MNIST', 'Fashion_MNIST', 'MNIST2_6']
nclasses = [2, 2, 10, 10, 2]
n = len(datasets)
n_features = [9, 8, 784, 784, 784]

for tt in range(n):
    binary = False
    if nclasses[tt] == 2:
        binary = True

    bst = xgb.Booster()
    model_path = '{}/models/rxgb/{}/{}_rxgb.model'.format(home, datasets[tt], datasets[tt])
    bst.load_model(model_path)
    test_data, test_label = load_svmlight_file('{}/chosen_sample/rxgb/{}_rxgb_samples.s'.format(home, datasets[tt]), n_features = n_features[tt])
    test_data = test_data.toarray()
    test_label = test_label.astype('int')
    if len(test_label) >= 1000:
        test_data = test_data[:500]
        test_label = test_label[:500]

    ori_points = []
    results = []
    for i in range(len(test_label)):
        s = test_data[i]
        sl = test_label[i]
        r = cheng_attack_rxgb.attack(bst, test_data, test_label, s, sl, nclasses, i)
        ori_points.append(s)
        results.append(r)
        print('{} is done'.format(i))

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
    pert['pert point'] = points
    pert['ori point'] = ori_points
    os.chdir('{}/attack/cheng'.format(home))
    pert.to_csv('{}_cheng_attack_rxgb.txt'.format(datasets[tt]))
    with open('{}_cheng_rxgb_ave.txt'.format(datasets[tt]), 'w') as f:
        f.write('average distance: ' + str(total_dis/len(test_label)))

    f.close()
    print('{} is done'.format(datasets[tt]))
    