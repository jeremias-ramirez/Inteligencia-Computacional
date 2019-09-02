# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:50:40 2019

@author: Messi
"""
import numpy as np

def trainning(trn,yd, w, vel):
    for j in range(len(trn[:,0])):
        z = sum(trn[j,:] * w[:])
        y = 1 if z >= 0 else -1
        error=yd[j] - y
        w = w + vel * error * trn[j,:]
    return w

