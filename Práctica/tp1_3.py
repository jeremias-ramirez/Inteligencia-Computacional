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

# poner fijo la semilla para que todos los pesos de distintas red tengan los mismo valores
np.random.seed(19680801)


reader = np.genfromtxt("files/concentlite.csv", delimiter=',')
trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
yd = np.expand_dims(reader[:, 2], axis=1)

##################
#caso tranformarlo a una dimension
mediaTotal = np.mean(reader[:, 0:2], axis = 0)# obtenga la media por cada eje

mediaTotal = mediaTotal.reshape((1,2)) # la transforma en una fila con dos columnas  
trnR = np.linalg.norm((reader[:, 0:2] - mediaTotal), axis = 1) # a cada elemento le resto la media (fila x fila) y luego obengo la distacia euclidea

#lo siguiente es agregar la entrada x0 = -1
trnR = np.expand_dims(trnR, axis = 1)
trnR = np.append(-np.ones((len(trnR[:,0]), 1)), trnR[:,:], 1)


# inicializo aleatoriamente los pesos W - vector de entradas y vector con la cantidad de neuronas
w = initW.initialize_w( np.ones((len(trn[0,:]),1)), np.array([5,1], np.int ))
w2 = w.copy()

wR = initW.initialize_w( np.ones((len(trnR[0,:]), 1)), np.array([1], np.int )) #inicializar los pesos de un perctron simple una capa con una neurana ([1])

vel = 0.05
velM = 0.3

epoc = 10


accurV = np.zeros((epoc,1))
errorV = np.zeros((epoc,1))

#con momento
accurVM = np.zeros((epoc,1))
errorVM = np.zeros((epoc,1))

#radial
accurVR = np.zeros((epoc,1))
errorVR = np.zeros((epoc,1))


tasa = 0.8


for i in range(epoc):
    deltaW = None
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis = 1)
        inputVR = np.expand_dims(trnR[j,:], axis = 1)

        y1 = salY.salidasy(inputV,w)
        y2 = salY.salidasy(inputV,w2)
        yR = salY.salidasy(inputVR,wR)

        w = bp.backpropagation(w,y1,yd[j],vel)
        w2,deltaW = bp.backpropagation_momento(w2,y2,yd[j],vel,velM,deltaW)
        wR = bp.backpropagation(wR,yR,yd[j],vel)

    accur = 0
    error = 0
    
    accurM = 0
    errorM = 0
  
    accurR = 0
    errorR = 0

    for j in range(len(trn[:,0])):

        inputV = np.expand_dims(trn[j,:], axis = 1)
        inputR = np.expand_dims(trnR[j,:], axis = 1)
    
        y = salY.salidasy(inputV, w)
        y2 = salY.salidasy(inputV, w2)
        yR = salY.salidasy(inputR, wR)

        ye = yd[j]
        ysalida = 1 if y[-1][-1] > 0 else -1
        ysalida2 = 1 if y2[-1][-1] > 0 else -1
        ysalidaR = 1 if yR[-1][-1] > 0 else -1

        accur = (accur + 1 if ysalida == ye else accur)
        accurM = (accurM + 1 if ysalida2 == ye else accurM)
        accurR = (accurR + 1 if ysalidaR == ye else accurR)

        error += (ye - y[-1][-1]) ** 2
        errorM += (ye - y2[-1][-1]) ** 2
        errorR += (ye - yR[-1][-1]) ** 2

    accurV[i] = accur/(np.size(trn[:,0]))
    errorV[i] = error/(np.size(trn[:,0]))

    accurVM[i] = accurM/(np.size(trn[:,0]))
    errorVM[i] = errorM/(np.size(trn[:,0]))
    
    accurVR[i] = accurR/(np.size(trn[:,0]))
    errorVR[i] = errorR/(np.size(trn[:,0]))


plt.figure("error cuadratico medio") 
plt.plot(range(epoc), errorV, 'k', range(epoc), errorVM, 'g', range(epoc), errorVR, 'b')

plt.figure("tasa de aciertos")
plt.plot(range(epoc), accurV, 'k', range(epoc), accurVM, 'g', range(epoc), accurVR, 'b')

#plt.show()
#
def showDistrib(data, w = None, trn = None):
    clase1= np.zeros((len(data[:,0]),3))
    clase2 = np.zeros((len(data[:,0]),3))
    for j in range(len(data[:,0])):
        inputV = np.expand_dims(data[j,:], axis = 1)
        if not (w == None):
            y = salY.salidasy(inputV, w)[-1][-1]
            if y > 0:
                clase1[j] = data[j,:]
                plt.scatter(clase1[j,1],clase1[j,2], c = "g")
            else:
                clase2[j] = data[j,:]
                plt.scatter(clase2[j,1],clase2[j,2], c = "r")

        else :
            y = data[j,-1]
            if y > 0:
                clase1[j] = data[j,:]
                plt.scatter(clase1[j,0],clase1[j,1], c = "g")
            else:
                clase2[j] = data[j,:]
                plt.scatter(clase2[j,0],clase2[j,1], c = "r")


plt.subplot(221)
showDistrib(reader)
plt.scatter(mediaTotal[0,0], mediaTotal[0,1], c = "k", marker = ">")
plt.subplot(222)
showDistrib(trn,w)

plt.subplot(223)
showDistrib(trn,w2)

plt.subplot(224)

for j in range(len(trnR[:,0])):
    inputV = np.expand_dims(trnR[j,:], axis = 1)
    y = salY.salidasy(inputV, wR)[-1][-1]
    if y > 0:
         plt.scatter(trn[j,1], trn[j,2], c = "g")
    else:
        plt.scatter(trn[j,1], trn[j,2], c = "r")



plt.show()
