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

for i in range(cantEntrenam):
    tupla = vc.getPartitions(data, 0.80)        
    dataTrain = data[tupla[1],:]
    w = np.random.uniform(-0.5,0.5,W)
    for j in range (epoc):
        w = trn.trainning(dataTrain, w, 0.2)
        desempeñoV = val.validation(dataTrain, w)
    
    dataTest = data[tupla[0],:]
    desempeñoP = val.validation(dataTest, w)

    accurV[i]=desempeñoP/len(dataTest[:,1])
    
plt.plot(range(0,cantEntrenam), accurV)
plt.show()