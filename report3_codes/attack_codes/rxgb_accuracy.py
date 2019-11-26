# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:14:27 2019

@author: Ying
"""


import xgboost as xgb
import os
from sklearn.datasets import load_svmlight_file
from scipy import sparse
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
    parser.add_argument('model_name', type=str, choices=['xgb','rxgb'], help = 'natural or robust model')
    parser.add_argument('testset', type = str, help='test data path')
    parser.add_argument('m', type = str, help='model path')
    parser.add_argument('c', type=int, help='number of classes')
    parser.add_argument('n', type=int, help='number of features')
    parser.add_argument('--feature_start', type=int, default=0, choices=[0,1], help='feature number starts from which index? For cod-rna and higgs, this should be 0.')

    args = parser.parse_args()
    d_name = args.data_name
    m_name = args.model_name
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

    if nclasses == 2:
        binary = True
    else:
        binary = False
        
    m = xgboost_wrapper(bst, binary)
    count = 0
    for i in range(test_data.shape[0]):
        s = test_data[i]
        pred = m.predict(s)
        if pred == test_label[i]:
            count+=1
    
    acc = count / len(test_label)
    print('accuracy on test set: {}%'.format(acc * 100))
    
    with open('{}_{}_ave.txt'.format(d_name, m_name), 'w') as f:
        f.write('accuracy on test set: {}%'.format(acc * 100))

    f.close()
    print('{} {} is done'.format(d_name, m_name))
    