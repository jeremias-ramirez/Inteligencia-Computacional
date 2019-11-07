import numpy as np
import sys
sys.path.append("../")

import trainning as trnD
from matplotlib import pyplot as plt


# poner fijo la semilla para que todos los pesos de distintas red tengan los mismo valores
np.random.seed(1568982731)

data1 = np.genfromtxt("files/irisbin.csv", delimiter=',')

data2 = np.genfromtxt("files/iris_k_medias_13.csv", delimiter=',')

vel = 0.2
velM = 0.5
epoc = 10
k = 30

lenIn_1 = 4
estrucRed1 = [[8, 6, 3], ["sigmoid", "sigmoid","sigmoid"]]

lenIn_2 = 13
estrucRed2 = [[3], ["linear"]]

print("Entrenamiento MLP")
results1 = trnD.trainningPar(data1, k, lenIn_1, estrucRed1, epocas = epoc, velM =velM)

print("Entrenamiento RBF")
results2 = trnD.trainningPar(data2, k, lenIn_2, estrucRed2, epocas = epoc, velM =velM)

figAcc = plt.figure(1)
figErr = plt.figure(2)
figEpc = plt.figure(3)

axAcc = figAcc.add_subplot(111)
axErr = figErr.add_subplot(111)
axEpc = figEpc.add_subplot(111)

axAcc.set_title("Curva % aciertos")
axErr.set_title("Curva error cuadratico medio")
axEpc.set_title("Epocas utilizadas por particion")

cantPar = len(results1)

axAcc.set_xticks(np.arange(0, cantPar))
axErr.set_xticks(np.arange(0, cantPar))
axEpc.set_xticks(np.arange(0, cantPar))

axAcc.set_xlabel(" Particiones ")
axErr.set_xlabel(" Particiones ")
axEpc.set_xlabel(" Particiones ")

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

axEpc.bar(range(0,cantPar), epocV1, 0.2)
axEpc.bar(np.arange(0,cantPar)+0.2, epocV2, 0.2)
axEpc.legend(("MLP", "RBF"), loc = "upper right")


plt.show()


