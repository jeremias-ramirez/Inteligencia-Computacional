"""
Created on Thu Aug 22 10:33:13 2019

"""

import numpy as np
from matplotlib import pyplot as plt
#import math
import trainning as trn
import validation as val
import val_cruzada_ej2 as vc

data = np.genfromtxt("spheres2d70.csv", delimiter = ",")
epoc = 200
cantEntrenam = 10
accurV=np.zeros((cantEntrenam,1))
tasa = 0.82

for i in range(cantEntrenam):
    tupla = vc.getPartitions(data, 0.80)        
    dataTrain = data[tupla[1],:]
    w = np.random.uniform(-0.5,0.5,W)
    for j in range (epoc):
        w = trn.trainning(dataTrain, w, 0.01)
        desempeñoV = val.validation(dataTrain, w)
        #print(desempeñoV)
        if tasa < desempeñoV:
            break
    
    print(j)
    
    dataTest = data[tupla[0],:]
    desempeñoP = val.validation(dataTest, w)

    accurV[i]=desempeñoP
     
plt.plot(range(0,cantEntrenam), accurV)
plt.show()