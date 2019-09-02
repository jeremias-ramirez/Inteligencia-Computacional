# -*- coding: utf-8 -*-

import csv
import numpy as np
from matplotlib import pyplot as plt
import trainning as train
import validation as val

def getPartitions(data, porcenTrain):
    H, W = data.shape
    indexs = np.arange(0, H, 1)
    
    porcenPrueba = round(1 - porcenTrain, 2)
    cantPrueba = int(H * porcenPrueba)
     
    np.random.shuffle(indexs)
    
    vPrueba = indexs[0:cantPrueba]
    vTrain = indexs[cantPrueba::]

    return vPrueba, vTrain

def getPartitionsk_fold(data,indexs,porcenTrain,k):
    H, W = data.shape
    porcenPrueba = round(1 - porcenTrain, 2)
    cantPrueba = int(H * porcenPrueba)
    vPrueba = indexs[k*cantPrueba:(k+1)*cantPrueba]
     
    c = [True if x not in vPrueba else False for x in indexs]
    vTrain=indexs[c]
    return vPrueba, vTrain

def k_fold(data,k,epoc,vel,tasa):
    H, W = data.shape
    accurV = np.zeros((k,1))
    porTest = 1/k
    porTrn = round(1 - porTest, 2)
    
    indexs = np.arange(0, H, 1)
    np.random.shuffle(indexs)

    
    for i in range(k):
        tupla = getPartitionsk_fold(data,indexs,porTrn,i)
        dataTrain = data[tupla[1],:]
        H, W = dataTrain.shape
        trn = np.append(-np.ones((len(dataTrain[:,1]),1)),dataTrain[:, 0:W-1],1)
        yd = dataTrain[:, W-1]
        w = np.random.uniform(-0.5,0.5,W)
            
        for j in range (epoc):
            w = train.trainning(trn,yd, w, vel)
            desempeñoV = val.validation(trn,yd, w)
            if desempeñoV > tasa:
                break
        
        dataTest = data[tupla[0],:]
       
        H, W = dataTest.shape
        test = np.append(-np.ones((len(dataTest[:,1]),1)),dataTest[:, 0:W-1],1)
        yd = dataTest[:, W-1]
    
        desempeñoP = val.validation(test,yd, w)
        accurV[i]=desempeñoP       

    plt.plot(range(0,k), accurV)
    plt.show()
    

data = np.genfromtxt("files/spheres1d10.csv", delimiter = ",")
k_fold(data,20,20,0.01,0.8)


