import numpy as np
import sys
sys.path.append("../")

import initialize_w as initW
import trainning as trnD
import partitions as prt

from matplotlib import pyplot as plt


# poner fijo la semilla para que todos los pesos de distintas red tengan los mismo valores
np.random.seed(12731)

data1 = np.genfromtxt("files/XOR_trn.csv", delimiter=',')

data2 = np.genfromtxt("files/xor_k_medias_4.csv", delimiter=',')

vel = 0.2
velM = 0.5
epoc = 1
k = 100

lenIn_1 = 2
estrucRed1 = [[2, 1], ["sigmoid", "sigmoid"]]


lenIn_2 = 4
estrucRed2 = [[1], ["linear"]]

print("Entrenamiento MLP")
results1 = trnD.trainningPar(data1, k, lenIn_1, estrucRed1, epocas = epoc, velM =velM)

print("Entrenamiento RBF")
results2 = trnD.trainningPar(data2, k, lenIn_2, estrucRed2, epocas = epoc, velM =velM)

figAcc = plt.figure(1)
figErr = plt.figure(2)

axAcc = figAcc.add_subplot(111)
axErr = figErr.add_subplot(111)

axAcc.set_title("Curva % aciertos")
axErr.set_title("Curva error cuadratico medio")

cantPar = len(results1)

axAcc.set_xticks(np.arange(0, cantPar))
axErr.set_xticks(np.arange(0, cantPar))

axAcc.set_xlabel(" Particiones ")
axErr.set_xlabel(" Particiones ")

accV1 = np.zeros((cantPar))
errV1 = np.ones((cantPar))
epocV1 = np.ones((cantPar))

accV2 = np.zeros((cantPar))
errV2 = np.ones((cantPar))
epocV2 = np.ones((cantPar))

for i in range(cantPar):
    accV1[i] = results1[i][1]
    accV2[i] = results2[i][1]
    
    errV1[i] = results1[i][2]
    errV2[i] = results2[i][2]
    
    epocV1[i] = results1[i][3]
    epocV2[i] = results2[i][3]



axAcc.plot(range(0,cantPar), accV1, range(0,cantPar), accV2)
axAcc.legend(("MLP", "RBF"), loc = "upper right")

axErr.plot(range(0,cantPar), errV1, range(0,cantPar), errV2)
axErr.legend(("MLP", "RBF"), loc = "upper right")


plt.show()


