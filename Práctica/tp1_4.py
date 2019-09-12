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
import time  

data = np.genfromtxt("files/irisbin.csv", delimiter=',')

trn = np.append(-np.ones((len(data[:,1]),1)), data[:, 0:4],1)
#yd = np.expand_dims(data[:, 4:], axis=1)

#np.random.seed(19680801)
# inicializo aleatoriamente los pesos W - vector de entradas y vector con la cantidad de neuronas


#w = initW.initialize_w( np.ones((5,1)), np.array([8,6,3], np.int ))
#w = initW.initialize_w( np.ones((5,1)), np.array([3, 3, 5, 3, 3], np.int ))

vel = 0.05
velM = 0.4

epoc = 200

tasa = 0.9

accurV = np.zeros((epoc,1))
errorCV = np.zeros((epoc,1))

k = 5

H, W = data.shape

cantPart = int(H/k)
accurVPar = np.zeros((cantPart,1))
errorVPar = np.zeros((cantPart,1))
errorVar = np.zeros((cantPart,1))
errorDesv = np.zeros((cantPart,1))
errorT = np.zeros((cantPart,1))

indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)
#vPrueba = np.zeros((int(H/k),1))
#vTrain = np.zeros((int(H/k),1))

for i in range(int(H/k)):
    tupla = partition.getPartitions_leave_k_out(indexs, k, i)
#    np.append(vPrueba, tupla[1])
#    np.append(vTrain, tupla[0])
    w = initW.initialize_w( np.ones((5,1)), np.array([8,6,3], np.int ))
    dataTrain = data[tupla[1], :]
    trn = np.append(-np.ones((len(dataTrain[:,1]),1)), dataTrain[:, 0:4], 1)
    yd = np.expand_dims(dataTrain[:, 4::], axis = 1)

    for e in range(epoc):
        deltaW = None
        for j in range(len(trn[:,0])):
            inputV = np.expand_dims(trn[j,:], axis = 1)
#            inputV = np.expand_dims(trn[j,:], axis=1)
            y1 = salY.salidasy(inputV, w)
#            for ys in y1:
#                print(ys.shape)
#            print(y1) 

            w,deltaW = bp.backpropagation_momento(w, y1, yd[j, :].T, vel, velM, deltaW)
   
        accur = 0
        errorC = 0
            
        for j in range(len(trn[:,0])):
            inputV = np.expand_dims(trn[j,:], axis=1)
            y = salY.salidasy(inputV, w)
        
            ye = yd[j, :].T
#            print(ye)
#            print(y[-1])
#            time.sleep(1)
            ysalida1 = 1 if y[-1][1] > 0 else -1
            ysalida2 = 1 if y[-1][2] > 0 else -1
            ysalida3 = 1 if y[-1][3] > 0 else -1
    
            accur = (accur + 1) if (ysalida1 == ye[0] and ysalida2 == ye[1] and ysalida3 == ye[2]) else accur
   

            errorC += np.linalg.norm(ye - y[-1][1:])**2
 #           errorC += np.linalg.norm(y[-1][1:] - ye)

        desempeño =accur/(np.size(trn[:,0]))
        accurV[e] = accur/(np.size(trn[:,0]))
        errorCV[e] = errorC/(np.size(trn[:,0]))
#        print(desempeño)
#        if desempeño>tasa:
#            break

#        print("accur: {}, errorC: {}".format(accurV[e], errorCV[e]))
    
    accurVPar[i] = np.mean(accurV)
    errorVPar[i] = np.mean(errorCV)
    errorVar[i] = np.var(errorCV)
    errorDesv[i] = np.std(errorCV)


#print(np.mean(accurV))
#plt.plot(range(epoc), accurV, 'k', range(epoc),errorCV, 'g')



######## PRUEBA
    dataTest = data[tupla[0], :]
    trn = np.append(-np.ones((len(dataTest[:,1]),1)), dataTest[:, 0:4], 1)
    yd = np.expand_dims(dataTest[:, 4::], axis = 1)
    accur = 0
    errorC = 0
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis = 1)
        y = salY.salidasy(inputV, w)
    
        ye = yd[j, :].T
        ysalida1 = 1 if y[-1][1] > 0 else -1
        ysalida2 = 1 if y[-1][2] > 0 else -1
        ysalida3 = 1 if y[-1][3] > 0 else -1
    
        accur = (accur + 1) if (ysalida1 == ye[0] and ysalida2 == ye[1] and ysalida3 == ye[2]) else accur
       
    
        errorC += np.linalg.norm(y[-1][1:] - ye)**2
    
    accur = accur/(np.size(trn[:,0]))
    errorC = errorC/(np.size(trn[:,0]))
    errorT[i] = errorC
    print("accur: {}, errorC: {}".format(accur, errorC))


plt.plot(range(cantPart), accurVPar, 'k')
plt.plot(range(cantPart), errorVPar, 'b', range(cantPart), errorT, 'g')
plt.show()    