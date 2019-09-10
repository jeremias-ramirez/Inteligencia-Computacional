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
w2 = w.copy()

vel = 0.05
velM = 0.3

epoc = 150

accurV = np.zeros((epoc,1))
errorV = np.zeros((len(trn[:,1]),epoc))

accurV2 = np.zeros((epoc,1))
errorV2 = np.zeros((len(trn[:,1]),epoc))


tasa = 0.8

errorCV = np.zeros((epoc,1))
errorCV2 = np.zeros((epoc,1))


for i in range(epoc):
    deltaW = None
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis=1)
        y1 = salY.salidasy(inputV,w)
        y2 = salY.salidasy(inputV,w2)

        w = bp.backpropagation(w,y1,yd[j],vel)

        w2,deltaW = bp.backpropagation_momento(w2,y2,yd[j],vel,velM,deltaW)

    accur = 0
    errorC = 0
    
    accur2 = 0
    errorC2 = 0

    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis=1)
        y = salY.salidasy(inputV, w)
        print(y)
        y2 = salY.salidasy(inputV, w2)

        ye = yd[j]
        ysalida = 1 if y[-1][-1] > 0 else -1
        ysalida2 = 1 if y2[-1][-1] > 0 else -1

        accur = (accur + 1 if ysalida == ye else accur)
        accur2 = (accur2 + 1 if ysalida2 == ye else accur2)

        errorC += (ye - y[-1][-1]) ** 2

        errorC2 += (ye - y2[-1][-1]) ** 2

    accurV[i] = accur/(np.size(trn[:,0]))
    errorCV[i] = errorC/(np.size(trn[:,0]))

    accurV2[i] = accur2/(np.size(trn[:,0]))
    errorCV2[i] = errorC2/(np.size(trn[:,0]))

    print("accur: {}, errorC: {}".format(accurV[i], errorCV[i]))
    #print(desempeño)
#    if desempeño>tasa:
#        break

print(np.mean(accurV))
plt.plot(range(epoc), accurV, 'k', range(epoc),errorCV, 'g')

plt.plot(range(epoc), accurV2,'b', range(epoc),errorCV2, 'r')

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

