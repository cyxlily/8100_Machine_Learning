# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:26:18 2019

@author: ycai
"""

import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
os.chdir('{}/codes'.format(home))
import norm_split
import numpy as np


datasets = ['breast_cancer', 'diabetes', 'ionosphere', 'covtype', 'cod_rna', 'cod_rna', 'HIGGS', 'Sensorless']
n = len(datasets)
labels = [10, 0, -1, -1, 0, 0, 0, -1]

file_name = ['breast-cancer-wisconsin.data', 'diabetes.txt', 'ionosphere.data', 'covtype.data', 'cod_rna_training.txt', 'cod_rna_test.txt', 'HIGGS.csv', 'Sensorless_drive_diagnosis.txt']
full_size = [546+137, 614+154, 281+70, 400000+181012, 59535, 271617, 10500000+500000, 48509+10000]
test_size = [137, 154, 70, 181012, 59535, 271617, 500000, 10000]
nfeatures = [9, 8, 34, 54, 8, 8, 28, 48]
column_format = [0, 1, 0, 0, 1, 1, 0, 0]
seporators = [',', ' ', ',', ',', ' ', ' ', ',', ' ']
headers = [None] * n
missing = ['?', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
drops = [[0], [], [], [], [], [], [], [], []]
seperated = [False, False, False, False, True, True, False,False] 

for i in range(n):
    norm_split.norm_split(datasets[i], labels[i], file_name[i], test_size[i], full_size[i], nfeatures[i], column_format[i], seporators[i], headers[i], missing[i], drops[i], seperated[i])
    print('{} is done'.format(datasets[i]))
    
   