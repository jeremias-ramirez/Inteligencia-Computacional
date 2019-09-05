# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:06:46 2019

@author: Shirli
"""

import numpy as np
import initialize_w as initW
import salidasy as salY
import backpropagation as bp
from matplotlib import pyplot as plt

reader = np.genfromtxt("files/concentlite.csv", delimiter=',')

trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
yd = np.expand_dims(reader[:, 2], axis=1)
np.random.seed(19680801)
# inicializo aleatoriamente los pesos W - vector de entradas y vector con la cantidad de neuronas
w = initW.initialize_w( np.ones((len(trn[0,:]),1)), np.array([5,1], np.int ))

vel = 0.1
epoc = 500

accurV = np.zeros((epoc,1))
wV = np.zeros((epoc,3))
errorV = np.zeros((len(trn[:,1]),epoc))
tasa = 0.8

errorCV = np.zeros((epoc,1))


for i in range(epoc):
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis=1)
        y = salY.salidasy(inputV,w)
        w = bp.backpropagation(w,y,yd[j],vel)
    accur = 0
    errorC = 0
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis=1)
        y = salY.salidasy(inputV, w)
        ye = yd[j]
        ysalida = 1 if y[-1][-1] > 0 else -1
        accur = (accur + 1 if ysalida == ye else accur)
        errorC += (ye - y[-1][-1]) ** 2
    accurV[i] = accur/(np.size(trn[:,0]))
    errorCV[i] = errorC/(np.size(trn[:,0]))
    print("accur: {}, errorC: {}".format(accurV[i], errorCV[i]))
    #print(desempeño)
#    if desempeño>tasa:
#        break

    accurV[i]=accur/len(trn[:,1])    

print(np.mean(accurV))
plt.plot(range(epoc), accurV, range(epoc),errorCV, 'g')

#plt.scatter(trn[:,1],trn[:,2])
plt.show()
clase1= np.zeros((len(trn[:,0]),3))
clase2 = np.zeros((len(trn[:,0]),3))
for j in range(len(trn[:,0])):
    inputV = np.expand_dims(trn[j,:], axis=1)
    y = salY.salidasy(inputV, w)
    if y[-1][-1] > 0:
        clase1[j] = trn[j,:]    
    else:
        clase2[j] = trn[j,:]    

plt.scatter(clase1[:,1],clase1[:,2], c="g")
plt.scatter(clase2[:,1],clase2[:,2], c = "r")

