# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:50:40 2019

"""

import numpy as np
import salidasy as salY
import backpropagation as bp

def trainningW(trn, yd, w, vel, velM = 0.0):
    deltaW = None
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis = 1)
        y = salY.salidasy(inputV, w)
        w, deltaW = bp.backpropagation_momento(w, y, yd[j], vel, velM, deltaW)
    return w
