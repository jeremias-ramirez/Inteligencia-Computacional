# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:22:40 2019

"""
import numpy as np
import salidasy as salY

def validation(val, yd, w):
    accur = 0
    errC = 0
    
    for j in range(len(val[:,0])):

        inputV = np.expand_dims(val[j,:], axis = 1)
    
        y = salY.salidasy(inputV, w)

        ye = yd[j].T
        fSigno = lambda x: 1 if x >= 0 else -1
        ysalida = np.array(list( map( fSigno, y[-1][1:])))

        accur = (accur + 1 if all(ysalida == ye) else accur)
        errC = np.linalg.norm(ye - y[-1][1:])**2
    
    N = np.size(val[:,0])

    return accur/N, errC/N

