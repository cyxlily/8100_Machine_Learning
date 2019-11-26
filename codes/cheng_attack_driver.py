# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:14:27 2019

@author: Ying
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
os.chdir('{}/codes'.format(home))
import cheng_attack
from sklearn.model_selection import train_test_split

datasets = ['breast_cancer', 'diabets', 'MNIST', 'Fashion_MNIST', 'MNIST2_6']
nclasses = [2, 2, 10, 10, 2]
n = len(datasets)

for i in range(n):
    model = xgb.Booster()
    model_path = '{}/models/xgb/{}_xgb.model'.format(home, datasets[i])
    model.load_model(model_path)
    test_df = pd.read_pickle('{}/chosen_sample/xgb/{}_xgb_samples.pkl'.format(home, datasets[i]))
    if test_df.shape[0] >= 1000:
        _, test_df = train_test_split(test_df, test_size = 200)


    test_df = test_df.reset_index(drop=True)
    test_data = np.array(test_df.drop(columns = ['label']))
    test_label = test_df['label'].tolist()
    dtest = xgb.DMatrix(test_data, label = test_label)

    ori_points = []
    results = []
    for tt in range(len(test_label)):
        s = test_data[tt]
        sl = test_label[tt]
        r = cheng_attack.attack(model, test_data, test_label, s, sl, nclasses, tt)
        ori_points.append(s)
        results.append(r)
        print('sample {} is done'.format(tt))


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
    pert.to_csv('{}_cheng_attack_xgb.txt'.format(datasets[i]))
    with open('{}_cheng_xgb_ave.txt'.format(datasets[i]), 'w') as f:
        f.write('average distance: ' + str(total_dis/len(test_label)))

    f.close()
    print('{} is done'.format(datasets[i]))