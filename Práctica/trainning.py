# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:50:40 2019

"""

import numpy as np
import salidasy as salY
import backpropagation as bp
import validation as valD
import partitions as prt
import initialize_w as initW
import multiprocessing as mp

def trainningPar(data, k, lenIn, estrucRed, epocas = 500, tasa = 0.95, vel = 0.1, velM = 0.0):
    
    H = data.shape[0]
    indexs = np.arange(0, H, 1)
    np.random.shuffle(indexs)
    
    cantPart = int(H/k)
    wV = np.zeros((cantPart, 1)) 
    accurV = np.zeros((cantPart, 1))
    errorV = np.zeros((cantPart, 1))
    
    pool = mp.Pool(processes = cantPart)
    argumentsL = list()

    for i in range(cantPart):
        w = initW.initialize_w( np.ones((lenIn + 1, 1)), np.array(estrucRed, np.int ))
        
        valInd, trnInd = prt.getPartitions_leave_k_out(indexs, k, i)
        
        dataTrain = data[trnInd, :]
        trn = np.append(-np.ones(((dataTrain[:, 1]).shape[0], 1)), dataTrain[:, 0:lenIn], 1)
        ydTrn = np.expand_dims(dataTrain[:, lenIn:], axis = 1)
        
        dataVal = data[valInd, :]
        val = np.append(-np.ones(((dataVal[:, 1]).shape[0], 1)), dataVal[:, 0:lenIn], 1)
        ydVal = np.expand_dims(dataVal[:, lenIn:], axis = 1)

        argumentsL.append((trn, val, ydTrn, ydVal, w, epocas, tasa, vel, velM))
    
    return pool.starmap(trainningEpoc, argumentsL)

def trainningEpoc(trn, val, ydTrn, ydVal, w, epocas = 500, tasa = 0.95, vel = 0.1, velM = 0.0):
    error = 0.0 
    epoCort = 0
    for epoCort in range(epocas):
        w = trainningW(trn, ydTrn, w, vel, velM)
        accur, error = valD.validation(val, ydVal, w)
        if tasa < accur:
            break
    return (w, accur, error, epoCort)

        
def trainningW(trn, yd, w, vel, velM = 0.0):
    deltaW = None
    for j in range(trn.shape[0]):
        inputV = np.expand_dims(trn[j,:], axis = 1)
        y = salY.salidasy(inputV, w)
        w, deltaW = bp.backpropagation_momento(w, y, yd[j], vel, velM, deltaW)
    return w
