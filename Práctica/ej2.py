# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:33:13 2019

@author: Messi
"""
import numpy as np
from matplotlib import pyplot as plt
import math
import trainning as trn
import validation as val
import val_cruzada_ej2 as vc

data = np.genfromtxt("spheres1d10.csv", delimiter = ",")

epoc = 10
cantEntrenam = 20
H, W = data.shape
accurV = np.zeros((cantEntrenam,1))
for i in range(cantEntrenam):
    tupla = vc.getPartition(data, 0.80)        
    dataTrain = data[tupla[1],:]
    w = np.random.uniform(-0.5,0.5,W)
    
    H, W = dataTrain.shape
    trn = np.append(-np.ones((len(dataTrain[:,1]),1)),dataTrain[:, 0:W-1],1)
    yd = dataTrain[:, W-1]

    for j in range (epoc):
        w = trn.trainning(trn,yd, w, 0.2)
        desempeñoV = val.validation(trn,yd, w)
    
    dataTest = data[tupla[0],:]
   
    H, W = dataTest.shape
    test = np.append(-np.ones((len(dataTest[:,1]),1)),dataTest[:, 0:W-1],1)
    yd = dataTest[:, W-1]

    desempeñoP = val.validation(test,yd, w)

    accurV[i]=desempeñoP/len(dataTest[:,1])
    
plt.plot(range(0,cantEntrenam), accurV)
plt.show()
