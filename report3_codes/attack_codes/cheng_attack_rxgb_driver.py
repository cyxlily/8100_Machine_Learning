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
from scipy import sparse
import time
import argparse
import numpy as np

class xgboost_wrapper():
	def __init__(self, model, binary=False):
		self.model = model 
		self.binary = binary
		#print('binary classification: ',self.binary)
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
		return test_predict[0]
	def predict_logits(self, input_data):
		input_data, _ = self.maybe_flat(input_data) 
		#input_back = np.copy(input_data)
		input_data = sparse.csr_matrix(input_data) 
		input_data = xgb.DMatrix(input_data) 
		test_predict = np.array(self.model.predict(input_data))
		return test_predict
	def predict_label(self, input_data):
		return self.predict(input_data)


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
    
    bst = xgb.Booster()
    bst.load_model(model_path)
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

    if nclasses == 2:
        binary = True
    else:
        binary = False
        
    m = xgboost_wrapper(bst, binary)
    ori_points = []
    results = []
    used_time = []
    count = 0
    tt = 0
    while count<n_selected:        
        s = test_data[tt]
        sl = test_label[tt]
        if m.predict(s) == sl:        
            start = time.time()
            r = cheng_attack_rxgb.attack(bst, test_data, test_label, s, sl, nclasses, tt)
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
    pert.to_csv('{}_cheng_attack_rxgb.txt'.format(d_name))
    with open('{}_cheng_rxgb_ave.txt'.format(d_name), 'w') as f:
        f.write('average distance: ' + str(total_dis/len(test_label)))
        f.write('\n\naverage used time: ' + str(np.mean(pert['used_time'])))

    f.close()
    print('{} is done'.format(d_name))
    