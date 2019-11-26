# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 08:48:29 2019

@author: ycai
"""

import os
curr = os.getcwd()

try:
    os.mkdir('{}/cai_cui'.format(curr))
except OSError:
    pass

os.chdir('{}/cai_cui'.format(curr))
home = os.getcwd()

folds = ['data', 'models','attack','chosen_sample', 'codes', 'test_acc']

for f in folds:
    try:
        os.mkdir('{}/{}'.format(home, f))
    except OSError:
        pass

os.chdir('{}/attack'.format(home))
try:
    os.mkdir('./cheng')
except OSError:
    pass
try:
    os.mkdir('./kattack')
except OSError:
    pass

folds = ['dt','xgb','rxgb']
os.chdir('{}/chosen_sample'.format(home))
for f in folds:
    try:
        os.mkdir('./{}'.format(f))
    except OSError:
        pass

os.chdir('{}/data'.format(home))
try:
    os.mkdir('./rxgb')
except OSError:
    pass

os.chdir('{}/models'.format(home))
for f in folds:
    try:
        os.mkdir('./{}'.format(f))
    except OSError:
        pass



