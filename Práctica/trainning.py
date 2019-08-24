# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:50:40 2019

@author: Messi
"""
import numpy as np
import val_cruzada_ej2 as vc

def trainning(data, w, vel):
    H, W = data.shape
    trn = np.append(-np.ones((len(data[:,1]),1)),data[:, 0:W-1],1)
    yd = data[:, W-1]
    
    for j in range(len(trn[:,0])):
        z = sum(trn[j,:] * w[:])
        y = 1 if z >= 0 else -1
        error=yd[j] - y
        w = w + vel * error * trn[j,:]

    return w

