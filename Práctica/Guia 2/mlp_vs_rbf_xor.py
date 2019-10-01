import numpy as np
import sys
sys.path.append("../")

import initialize_w as initW
import trainning as trnD
import partitions as prt

from matplotlib import pyplot as plt
import multiprocessing as mp


# poner fijo la semilla para que todos los pesos de distintas red tengan los mismo valores
np.random.seed(1568982731)

data1 = np.genfromtxt("files/XOR_trn.csv", delimiter=',')

data2 = np.genfromtxt("files/xor_k_medias_4.csv", delimiter=',')

vel = 0.2
velM = 0.5
epoc = 1
k = 500

lenIn_1 = 2
estrucRed1 = [2, 1]

lenIn_2 = 4
estrucRed2 = [1]

results1 = trnD.trainningPar(data1, k, lenIn_1, estrucRed1, epocas = epoc, velM =velM)

results2 = trnD.trainningPar(data2, k, lenIn_2, estrucRed1, epocas = epoc, velM =velM)

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


#results2 = trnD.trainningPar(data2, k, lenIn_2, estrucRed1, epocas = epoc, velM =velM)

#print(results1)
#print(results2)

#cantPart = int(H/k)
#
#accurV = np.zeros((cantPart,1))
#errorV = np.zeros((cantPart,1))
#
#pool = mp.Pool(processes = cantPart)
#
#argumentsL = list()
#for i in range(cantPart):
#    w = initW.initialize_w( np.ones((3, 1)), np.array([2, 1], np.int ))
#    
#    valInd, trnInd = prt.getPartitions_leave_k_out(indexs, k, i)
#    
#    dataTrain = data[trnInd, :]
#    trn = np.append(-np.ones(((dataTrain[:, 1]).shape[0], 1)), dataTrain[:, 0:2], 1)
#    
#    dataVal = data[valInd, :]
#    val = np.append(-np.ones(((dataVal[:, 1]).shape[0], 1)), dataVal[:, 0:2], 1)
#
#    yd = np.expand_dims(dataTrain[:, 2], axis = 1)
#    
#    argumentsL.append((trn, val, yd, w, epoc, 0.95, vel, velM))
#
#results = pool.starmap(trnD.trainningEpoc, argumentsL)

#for process in processes:
#    process.start()
#    #results.append(pool.apply(trnD.trainningEpoc, (trn, val, yd, w, epoc, 0.95, vel, velM,)))
#
#for process in processes:
#   process.join()
#
#results = [resultsQ.get() for process in processes] 

#for i in range(cantPart):
#
#    accurV[i] = results[i][1] 
#    errorV[i] = results[i][2] 
#
#
#
#plt.figure("error cuadratico medio") 
#plt.plot(range(cantPart), errorV)
#
#plt.figure("tasa de aciertos")
#plt.plot(range(cantPart), accurV)
#
#
#plt.show()

#w = initW.initialize_w( np.ones((lenIn + 1, 1)), np.array(estrucRed, np.int ))
#
#
#dataTrain = data[200:, :]
#trn = np.append(-np.ones(((dataTrain[:, 1]).shape[0], 1)), dataTrain[:, 0:lenIn], 1)
#
#dataVal = data[0:200, :]
#val = np.append(-np.ones(((dataVal[:, 1]).shape[0], 1)), dataVal[:, 0:lenIn], 1)
#
#ydTrn = np.expand_dims(dataTrain[:, lenIn:], axis = 1)
#
#ydVal = np.expand_dims(dataVal[:, lenIn:], axis = 1)
#
#result = trnD.trainningEpoc(trn, val, ydTrn, ydVal, w, epoc, velM = velM)
#print(result)







