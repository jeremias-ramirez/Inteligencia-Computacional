# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:06:46 2019

@author: Shirli
"""

import numpy as np
import initialize_w as initW
import salidasy as salY
import trainning as trnD
import validation as valD

import backpropagation as bp
from matplotlib import pyplot as plt

# poner fijo la semilla para que todos los pesos de distintas red tengan los mismo valores
np.random.seed(19680801)


reader = np.genfromtxt("files/XOR_trn.csv", delimiter=',')
trn = np.append(-np.ones(((reader[:,1]).shape[0], 1)), reader[:, 0:2], 1)
yd = np.expand_dims(reader[:, 2], axis=1)

# inicializo aleatoriamente los pesos W - vector de entradas y vector con la cantidad de neuronas
w = initW.initialize_w( np.ones(((trn[0,:]).shape[1], 1)), np.array([2,1], np.int ))

vel = 0.05
velM = 0.3

epoc = 300


accurV = np.zeros((epoc,1))
errorV = np.zeros((epoc,1))

tasa = 0.8


for i in range(epoc):
    w = trnD.trainningW(trn, yd, w, vel)
    
    accur, error = valD.validation(trn, yd, w)

    accurV[i] = accur
    errorV[i] = error

plt.figure("error cuadratico medio") 
plt.plot(range(epoc), errorV, 'k', range(epoc))

plt.figure("tasa de aciertos")
plt.plot(range(epoc), accurV, 'k', range(epoc))

plt.show()
