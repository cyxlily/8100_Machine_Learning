# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:00:44 2019

@author: ycai
"""

import numpy as np
import xgboost as xgb
import scipy


def predict(model, sampleX, nclasses):
    dtest = xgb.DMatrix(scipy.sparse.csr_matrix(sampleX))
    prediction = model.predict(dtest)
    if nclasses == 2:
        if prediction<0.5:
            prediction = 0
        else:
            prediction = 1
    else:
        prediction = int(prediction)
    return prediction



def g_theta_local(model, sampleX, sample_label, theta, pre_v, nclasses, ratio = 0.02, tolerance = 0.0005):
    theta = theta / np.linalg.norm(theta)
    if predict(model, (sampleX + theta * pre_v), nclasses) == sample_label:
        v_left = pre_v
        v_right = (1+ratio) * pre_v
        while predict(model, (sampleX + theta * v_right), nclasses) == sample_label:
            v_right = (1+ratio) * v_right          
            if v_right > 20:
                return float('inf'), float('inf')
    else:
        v_right = pre_v
        v_left = (1-ratio) * pre_v
        while predict(model, (sampleX + theta * v_left), nclasses) != sample_label:
            v_left = (1-ratio) * v_left
            if v_left <= tolerance*4:
                return float('inf'), float('inf')
    while (v_right - v_left) > tolerance:
        v_mid = (v_right + v_left) / 2
        if predict(model, (sampleX + v_mid * theta), nclasses) == sample_label:
            v_left = v_mid
        else:
            v_right = v_mid
    t = theta * v_right
    dis = np.abs(max(t, key=abs))
    return v_right, dis

def attack(model, tdata, tlabel, x0, y0, nclasses, index, step = 0.2, beta = 0.01, iterations = 1000):    
    q = 20    
    nf = len(x0)    
    best_theta, g_theta, dis = None, float('inf'), float('inf')
    for i in range(len(tlabel)):
        if predict(model, tdata[i], nclasses) != y0:
            theta = tdata[i] - x0
            initial_lbd = 1.0            
            lbd, distance = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, nclasses)
            if distance < dis:
                best_theta, g_theta, dis = theta, lbd, distance
    theta = best_theta
    pre_v = g_theta
    stopping = 0.0001    
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
        print(str(min_dis))
        if replaced:
            theta = min_theta
            pre_v = min_v
            count = 0
        else:
            count += 1
            #print('count: ' + str(count))
            if (count > 30):
                break
    return (index, min_dis, (x0 + min_theta * min_v))



def collect_result(result):
    global results
    results.append(result)


def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, nclasses, tolerance = 0.0001):
    lbd = initial_lbd
    lbd_hi = lbd
    lbd_lo = 0.0
    while (lbd_hi - lbd_lo) > tolerance:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        if predict(model, x0 + lbd_mid*theta, nclasses) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    t = theta * lbd_hi
    dis = np.abs(max(t, key=abs))
    return lbd_hi, dis

