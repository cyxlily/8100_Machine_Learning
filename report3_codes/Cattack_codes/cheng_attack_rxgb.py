# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:41:28 2019

@author: Ying
"""

import numpy as np
import xgboost as xgb
from scipy import sparse


class xgboost_wrapper():
	def __init__(self, model, binary=False):
		self.model = model 
		self.binary = binary
		#print('binary classification: ',self.binary)

	def maybe_flat(self, input_data):
		if not isinstance(input_data,np.ndarray):
			#print(type(input_data))
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
		input_back = np.copy(input_data)
		input_data = sparse.csr_matrix(input_data) 
		input_data = xgb.DMatrix(input_data) 
		test_predict = np.array(self.model.predict(input_data))
		return test_predict

	def predict_label(self, input_data):
		return self.predict(input_data)



def g_theta_local(model, sampleX, sample_label, theta, pre_v, nclasses, ratio = 0.02, tolerance = 0.001):
    theta = theta / np.linalg.norm(theta)
    if model.predict((sampleX + theta * pre_v)) == sample_label:
        v_left = pre_v
        v_right = (1+ratio) * pre_v
        while model.predict( (sampleX + theta * v_right)) == sample_label:
            v_right = (1+ratio) * v_right          
            if v_right > 20:
                return float('inf'), float('inf')
    else:
        v_right = pre_v
        v_left = (1-ratio) * pre_v
        while model.predict((sampleX + theta * v_left)) != sample_label:
            v_left = (1-ratio) * v_left
            if v_left <= tolerance*4:
                return float('inf'), float('inf')
    while (v_right - v_left) > tolerance:
        v_mid = (v_right + v_left) / 2
        if model.predict((sampleX + v_mid * theta)) == sample_label:
            v_left = v_mid
        else:
            v_right = v_mid
    t = theta * v_right
    dis = np.abs(max(t, key=abs))
    return v_right, dis

def attack(bst, tdata, tlabel, x0, y0, nclasses, index, step = 0.2, beta = 0.01, iterations = 1000):    
    if nclasses == 2:
        binary = True
    else:
        binary = False
    model = xgboost_wrapper(bst, binary=binary)
    q = 20    
    nf = len(x0)    
    best_theta, g_theta, dis = None, float('inf'), float('inf')
    for i in range(len(tlabel)):
        if model.predict(tdata[i]) != y0:
            theta = tdata[i] - x0
            initial_lbd = 1.0            
            lbd, distance = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, nclasses)
            if distance < dis:
                best_theta, g_theta, dis = theta, lbd, distance
    theta = best_theta
    pre_v = g_theta
    stopping = 0.0003    
    min_dis = dis
    min_theta = theta
    min_v = pre_v
    count = 0
    for t in range(iterations):        
        #print(str(t))
        grad = np.zeros(nf)
        for _i in range(q):
            u = np.random.normal(size = nf)            
            g1, _ = g_theta_local(model, x0, y0, theta + u * beta, pre_v, nclasses)
            if g1 <= 20: 
                grad = grad + (g1-pre_v)/beta * u            
        u = 1.0/q * grad        
        replaced = False
        new_step = step
        for _i in range(15):
            #print('_i')            
            new_theta = theta - u * new_step
            new_v, new_dis = g_theta_local(model, x0, y0, new_theta, pre_v, nclasses)
            if min_dis - new_dis > stopping:
                replaced = True
                min_dis = new_dis
                min_theta = new_theta
                min_v = new_v
                new_step = new_step + step
                #print('replaced')
            else:
                break
        new_step = step
        for _i in range(15):
            #print('while')
            new_step = new_step * 0.5
            new_theta = theta - u * new_step
            new_v, new_dis = g_theta_local(model, x0, y0, new_theta, pre_v, nclasses)
            if min_dis - new_dis > stopping:
                replaced = True
                min_dis = new_dis
                min_theta = new_theta
                min_v = new_v    
                #print('replaced')       
            else:
                break
        #print(str(min_dis))
        if replaced:
            theta = min_theta
            pre_v = min_v
            count = 0
        else:
            count += 1
            #print('count: ' + str(count))
            if (count > 30):
                break
        if min_dis < 0.001:
            break
    return (index, min_dis, (x0 + min_theta * min_v))



def collect_result(result):
    global results
    results.append(result)


def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, nclasses, tolerance = 0.001):
    lbd = initial_lbd
    lbd_hi = lbd
    lbd_lo = 0.0
    while (lbd_hi - lbd_lo) > tolerance:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        if model.predict( x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    t = theta * lbd_hi
    dis = np.abs(max(t, key=abs))
    return lbd_hi, dis

