# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:22:28 2019

@author: Shirli
"""

import numpy as np
import initialize_w as initW
import salidasy as salY
import backpropagation as bp
import partitions as partition
from matplotlib import pyplot as plt


data = np.genfromtxt("files/irisbin.csv", delimiter=',')

trn = np.append(-np.ones((len(data[:,1]),1)), data[:, 0:4],1)
#yd = np.expand_dims(data[:, 4:], axis=1)

np.random.seed(19680801)
# inicializo aleatoriamente los pesos W - vector de entradas y vector con la cantidad de neuronas

w = initW.initialize_w( np.ones((5,1)), np.array([5,3], np.int ))

vel = 0.05

epoc = 10

accurV = np.zeros((epoc,1))
errorV = np.zeros((len(trn[:,1]),epoc))

tasa = 0.8

errorCV = np.zeros((epoc,1))

k = 10
H, W = data.shape
indexs = np.arange(0, H, 1)
#vPrueba = np.zeros((int(H/k),1))
#vTrain = np.zeros((int(H/k),1))

for i in range(int(H/k)):
    tupla = partition.getPartitions_leave_k_out(indexs, k, i)
#    np.append(vPrueba, tupla[1])
#    np.append(vTrain, tupla[0])
    dataTrain = data[tupla[1], :]
    trn = np.append(-np.ones((len(dataTrain[:,1]),1)), dataTrain[:, 0:4], 1)
    yd = np.expand_dims(dataTrain[:, 4::], axis = 1)

    for e in range(epoc):

        for j in range(len(trn[:,0])):
            inputV = np.expand_dims(trn[j,:], axis = 1)
#            inputV = np.expand_dims(trn[j,:], axis=1)
            y1 = salY.salidasy(inputV, w)
            for ys in y1:
                print(ys.shape)
            print(y1) 

            w = bp.backpropagation(w, y1, yd[j, :].T, vel)
   
            accur = 0
            errorC = 0
            
        for j in range(len(trn[:,0])):
#            inputV = np.expand_dims(trn[j,:], axis=1)
            y = salY.salidasy(inputV, w)
        
            ye = yd[j, :].T
            ysalida1 = 1 if y[-1][-3] > 0 else -1
            ysalida2 = 1 if y[-1][-2] > 0 else -1
            ysalida3 = 1 if y[-1][-1] > 0 else -1
    
            accur = (accur + 1) if (ysalida1 == ye[0] and ysalida2 == ye[1] and ysalida3 == ye[2]) else accur
   
            errorC += (ye[0] - ysalida1 + ye[1] - ysalida2 + ye[2] - ysalida3) ** 2
        
        accurV[i] = accur/(np.size(trn[:,0]))
        errorCV[i] = errorC/(np.size(trn[:,0]))
    
    
        print("accur: {}, errorC: {}".format(accurV[i], errorCV[i]))
        #print(desempeño)
    #    if desempeño>tasa:
    #        break
#
#print(np.mean(accurV))
#plt.plot(range(epoc), accurV, 'k', range(epoc),errorCV, 'g')
#
#plt.plot(range(epoc), accurV2,'b', range(epoc),errorCV2, 'r')
#
##plt.scatter(trn[:,1],trn[:,2])
#plt.show()
#clase1= np.zeros((len(trn[:,0]),3))
#clase2 = np.zeros((len(trn[:,0]),3))
#for j in range(len(trn[:,0])):
#    inputV = np.expand_dims(trn[j,:], axis=1)
#    y = salY.salidasy(inputV, w)
#    if y[-1][-1] > 0:
#        clase1[j] = trn[j,:]    
#    else:
#        clase2[j] = trn[j,:]    
#
#plt.scatter(clase1[:,1],clase1[:,2], c="g")
#plt.scatter(clase2[:,1],clase2[:,2], c = "r")
#

