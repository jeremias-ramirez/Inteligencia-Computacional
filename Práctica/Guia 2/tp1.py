# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:53:18 2019

@author: Shirli
"""
import numpy as np
from matplotlib import pyplot as plt

reader = np.genfromtxt("files/xor_trn.csv", delimiter=',')

epoc = 100
tasa = 0.69

trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
result = reader[:, 2]

H, W = trn.shape
indexs = np.arange(0, H, 1)

k = 4  
#centroide = np.random.uniform(index,H,4)
w = np.random.uniform(-0.5,0.5,4)

### INICIALIZACION ALEATORIA DE LOS Cj(0)
aux = int(H/k)
np.random.shuffle(indexs)

c = np.zeros((k, aux))
centroide = np.zeros((k,1))

for i in range(k):
    c[i,:] = indexs[(i*aux):(i*aux)+aux]

ep = 0
flag= 1
P = np.zeros((k, aux))   
while ep <epoc:
    for s in range(k):
        for p in range(aux):
            if P[s, p] != c[s, p]:
                break
    
    for j in range(k):
        centroide[j] = (1/len(c[j])) * sum(c[j,:])

    P = np.zeros((k, aux))   
    Paux = np.zeros((aux))

    dist = np.zeros(k)
    cont = 0
    for m in range(H):
        minimoval = 19999999999999999999999999999
        for j in range(k):
            dist[j] = sum((trn[indexs[m],:] - centroide[j])* (trn[indexs[m],:] - centroide[j]))
            if minimoval > dist[j]:
                minimoval = dist[j]
                minimo = j
#            print(dist[j], j)
#        data = np.where((c[minimo] == indexs[m]))
#        if len(data[0]) == 0:
#            cont += 1
        Paux = P[minimo]
        np.append(Paux,indexs[m])
        P[minimo] = Paux
        
    c[:] = P[:]
#        print(c2)
#    c = c2[:,:]    
    ep += 1
      
