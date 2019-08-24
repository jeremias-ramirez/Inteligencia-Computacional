# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:22:40 2019

@author: Messi
"""
import numpy as np

def validation(data, w):
    H, W = data.shape
    trn = np.append(-np.ones((len(data[:,1]),1)),data[:, 0:W-1],1)
    yd = data[:, W-1]
    accur = 0
    
    for j in range(len(trn[:,0])):
        z = sum(trn[j,:] * w[:])
        y = 1 if z >= 0 else -1
        error=yd[j] - y
        accur = (accur + 1 if error == 0 else accur)
    return accur
