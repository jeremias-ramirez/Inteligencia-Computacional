# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt

reader = np.genfromtxt("or_trn.csv", delimiter=',')

w = np.random.uniform(-0.5,0.5,3)
epoc = 10

trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
result = reader[:, 2]
vel = 0.1
accurV=np.zeros((epoc,1))
wV = np.zeros((epoc,3))

errorV = np.zeros((len(trn[:,1]),epoc))
for i in range(epoc):
    for j in range(len(trn[:,0])):
        z = sum(trn[j,:] * w[:])
        y = 1 if z >= 0 else -1
        error=result[j] - y
        #errorV[j] = error
        w = w + vel * error * trn[j,:]
        wV[i] = w
        
    accur = 0
    for j in range(len(trn[:,0])):
        z = sum(trn[j,:] * w[:])
        y = 1 if z >= 0 else -1
        error=result[j] - y
        accur = (accur + 1 if error == 0 else accur)
    accurV[i]=accur/len(trn[:,1])    
                

plt.scatter(trn[:,1],trn[:,2])
plt.plot(trn[:,1],-trn[:,1]*w[1]/w[2]+w[0]/w[2], 'g')
plt.show()

reader = np.genfromtxt("or_tst.csv", delimiter=',')
trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
result = reader[:, 2]
accur = 0
for j in range(len(trn[:,0])):
    z = sum(trn[j,:] * w[:])
    y = 1 if z >= 0 else -1
    error=result[j] - y
    accur = (accur + 1 if error == 0 else accur)

print(accur/len(trn[:,1]))    

