# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:07:24 2019

@author: Ying
"""

import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import os
curr = os.getcwd()
find = curr.find('cai_cui')
home = curr[:find+7]
import pandas as pd
import random

class xgboost_wrapper():
	def __init__(self, model, binary=False):
		self.model = model 
		self.binary = binary
	def maybe_flat(self, input_data):
		if not isinstance(input_data,np.ndarray):
			print(type(input_data))
			input_data = np.copy(input_data.numpy())
		shape = input_data.shape
		if len(input_data.shape) == 1:
			input_data = np.copy(input_data[np.newaxis,:])
		if len(input_data.shape) >= 3:
			input_data = np.copy(input_data.reshape(shape[0],np.prod(shape[1:])))
		return input_data, shape
	def predict(self, input_data):
		input_data, _ = self.maybe_flat(input_data)
		ori_input = np.copy(input_data)
		np.clip(input_data, 0, 1, input_data)
		input_data = xgb.DMatrix(sparse.csr_matrix(input_data)) 
		ori_input = xgb.DMatrix(sparse.csr_matrix(ori_input))
		test_predict = np.array(self.model.predict(input_data))
		if self.binary:
			test_predict = (test_predict > 0.5).astype(int)
		else:
			test_predict = test_predict.astype(int) 
		return test_predict
	def predict_logits(self, input_data):
		input_data, _ = self.maybe_flat(input_data) 
		input_back = np.copy(input_data)
		input_data = sparse.csr_matrix(input_data) 
		input_data = xgb.DMatrix(input_data) 
		test_predict = np.array(self.model.predict(input_data))
		return test_predict
	def predict_label(self, input_data):
		return self.predict(input_data)

datasets = datasets = ['breast_cancer', 'diabetes', 'covtype', 'cod_rna', 'ijcnn1', 'Sensorless', 'webspam', 'MNIST', 'Fashion_MNIST', 'MNIST2_6']
n = len(datasets)
nclasses = [2, 2,  7,  2, 2, 11, 2, 10, 10, 2]
sample_size = [100, 100, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 1000]
n_features = [ 9, 8, 54, 8, 22, 48, 254, 784, 784, 784]

accuracy = []
train_accu = []
for i in range(n):
    binary = False
    if nclasses[i] == 2:
        binary = True
    feature_zbased = False
    if 'HIGGS' in datasets[i] or 'rna' in datasets[i]:
        feature_zbased = True
    model_name = '{}/models/rxgb/{}/{}_rxgb.model'.format(home, datasets[i], datasets[i])
    file_name = '{}/data/rxgb/{}_test.svm'.format(home, datasets[i])
    bst = xgb.Booster()
    bst.load_model(model_name)
    binary = binary
    model = xgboost_wrapper(bst, binary=binary)	
    test_data, test_labels = load_svmlight_file(file_name, n_features[i])
    test_data = test_data.toarray()
    test_labels = test_labels.astype('int')
    y = model.predict(test_data)
    temp = pd.DataFrame()
    temp['true'] = test_labels
    temp['pred'] = y
    correct = 0
    correct_classified = []
    for j in range(temp.shape[0]):
        if temp.iloc[j]['true'] == temp.iloc[j]['pred']:
            correct = correct + 1
            correct_classified.append(j)           
    selected = random.sample(correct_classified, min(sample_size[i], correct))
    accu = correct / temp.shape[0]
    accuracy.append(accu)
    X = test_data[selected, :]
    y = test_labels[selected]
    dump_svmlight_file(X, y, '{}/chosen_sample/rxgb/{}_rxgb_samples.s'.format(home, datasets[i]))
    
    file_name = '{}/data/rxgb/{}_train.svm'.format(home, datasets[i])
    train_data, train_labels = load_svmlight_file(file_name, n_features[i])
    train_data = train_data.toarray()
    train_labels = train_labels.astype('int')
    y = model.predict(train_data)
    temp = pd.DataFrame()
    temp['true'] = train_labels
    temp['pred'] = y
    correct = 0    
    for k in range(temp.shape[0]):
        if temp.iloc[k]['true'] == temp.iloc[k]['pred']:
            correct = correct + 1    
    accu = correct / temp.shape[0]
    train_accu.append(accu)
    print('{} is done'.format(datasets[i]))
    
acc_df = pd.DataFrame()
acc_df['dataset'] = datasets
acc_df['accuracy'] = accuracy
acc_df['train_accu'] = train_accu
os.chdir('{}/test_accu'.format(home))
acc_df.to_csv('rxgb_test_accu.txt')
acc_df.to_excel('rxgb_test_accu.xlsx')
