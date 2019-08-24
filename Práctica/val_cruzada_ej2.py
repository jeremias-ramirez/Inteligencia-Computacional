# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
from random import shuffle

def getPartitions(data, porcenTrain):
    H, W = data.shape
    indexs = np.arange(0, H, 1)
    
    porcenPrueba = round(1 - porcenTrain, 1)
    cantPrueba = int(H * porcenPrueba)
    
    np.random.shuffle(indexs)
    
    vPrueba = indexs[0:cantPrueba]
    vTrain = indexs[cantPrueba::]

    return vPrueba, vTrain

#data = np.genfromtxt("spheres1d10.csv", delimiter = ",")
#tupla = getPartitions(data, 0.80)