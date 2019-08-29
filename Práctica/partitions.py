# -*- coding: utf-8 -*-

import csv
import numpy as np

def getPartitions(data, porcenTrain):
    H, W = data.shape
    indexs = np.arange(0, H, 1)
    
    porcenPrueba = round(1 - porcenTrain, 1)
    cantPrueba = int(H * porcenPrueba)
    
    np.random.shuffle(indexs)
    
    vPrueba = indexs[0:cantPrueba]
    vTrain = indexs[cantPrueba::]

    return vPrueba, vTrain

def getPartitionsk_fold(data, porcenTrain,k):
    H, W = data.shape
    indexs = np.arange(0, H, 1)
    
    porcenPrueba = round(1 - porcenTrain, 1)
    cantPrueba = int(H * porcenPrueba)
    
    vPrueba = indexs[k*cantPrueba:(k+1)*cantPrueba]
     
    vTrain = indexs[!(k*cantPrueba:(k+1)*cantPrueba)]
    return vPrueba, vTrain

def k_fold(data,k,epoc):
    H, W = data.shape
    accurV = np.zeros((k,1))
    porTest = 1/k
    porTrn = round(1 - porTest, 1)
    tupla = getPartitions(data, porTrn)

    dataTrain = data[tupla[1],:]
    H, W = dataTrain.shape
    trn = np.append(-np.ones((len(dataTrain[:,1]),1)),dataTrain[:, 0:W-1],1)
    yd = dataTrain[:, W-1]

    for i in range(k):
        w = np.random.uniform(-0.5,0.5,W)
            
        for j in range (epoc):
            w = train.trainning(trn,yd, w, 0.2)
            desempeñoV = val.validation(trn,yd, w)
        
        dataTest = data[tupla[0],:]
       
        H, W = dataTest.shape
        test = np.append(-np.ones((len(dataTest[:,1]),1)),dataTest[:, 0:W-1],1)
        yd = dataTest[:, W-1]
    
        desempeñoP = val.validation(test,yd, w)
        accurV[i]=desempeñoP/len(dataTest[:,1])
        
    plt.plot(range(0,k), accurV)
    plt.show()
#data = np.genfromtxt("spheres1d10.csv", delimiter = ",")
#tupla = getPartitions(data, 0.80)
