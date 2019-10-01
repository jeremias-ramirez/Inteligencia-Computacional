# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:45:16 2019

@author: Shirli
"""
import numpy as np
from matplotlib import pyplot as plt

def entorno(fila, columna, paso_horizontal, paso_vertical, ra_vecindad, hn, wn):
    flag = True
    filaI = fila
    columnaI = columna
    indices = list()
    while flag:
        filaI = filaI + paso_vertical
        columnaI = columnaI + paso_horizontal 
        if filaI >= hn or columnaI >= wn:
            break
        if filaI < 0 or columnaI < 0 :
            break
        distancia = np.linalg.norm(np.array([filaI - fila, columnaI - columna]))
        if distancia > ra_vecindad:
            break
        else:
            indices.append((filaI,columnaI)) 
    return indices

def actualizaVecinos(listaVecinos, w, vel, datai):
    for i in range(len(listaVecinos)):
        columna = listaVecinos[i][1]
        fila = listaVecinos[i][0]
        w[fila,columna] = w[fila,columna] + vel * (datai-w[fila,columna])
    return w

def som(data, hn, wn):
    ra_vecindad = 0
    H, W = data.shape
    w = np.zeros((hn,wn,W))
    indexsH = np.arange(0, H, 1)
    indexsW = np.arange(0, W, 1)
    np.random.shuffle(indexsH)
    np.random.shuffle(indexsW)   
    vel = 0.01
    
    w_activ = np.zeros((H,1))
    epoc = 1
    
    for i in range(hn):
        for j in range(wn):
#            w[i,j,:] = data[indexsH[i]]
             w[i,j,:] = data[np.random.randint(0, H)]
#    print(w)
#    print(w.reshape((hn*wn, W))
    for ep in range(epoc):         
        for i in range(H):
    #        print(data[i,:])
            distancias = np.linalg.norm((data[i, :] - w.reshape((hn*wn, W) )), axis = 1,ord=2)
    #        print(distancias)
        
            ind = np.argmin(distancias, axis = 0)
#            print(ind)
            columna = ind%wn
            fila = int(ind/wn)
            w_activ[i]= int(ind)
            # actual
    #        print("fila, columna {} {}".format(fila, columna))
            w[fila,columna] = w[fila,columna] + vel * (data[i,:]-w[fila,columna])
           
            #vertical
            verticalesS = entorno(fila, columna, 0, 1, ra_vecindad, hn, wn)        
    #        print("verticalesS {}".format(verticalesS))
            verticalesN = entorno(fila, columna, 0, -1, ra_vecindad, hn, wn)        
    #        print("verticalesN {}".format(verticalesN))
    
    #        print("W viejo {}, verticalesS {}, verticalesN {}".format(w, verticalesS, verticalesN))
            
            w = actualizaVecinos(verticalesS, w, vel, data[i,:])
            w = actualizaVecinos(verticalesN, w, vel, data[i,:])        
    
    #        print("W actualizado {}".format(w))
            #horizontal
            horizontalesE = entorno(fila, columna, 1, 0, ra_vecindad, hn, wn)        
    #        print("horizontalesE {}".format(horizontalesE))
            horizontalesO = entorno(fila, columna, -1, 0, ra_vecindad, hn, wn)        
    #        print("horizontalesO {}".format(horizontalesO))
            w = actualizaVecinos(horizontalesE, w, vel, data[i,:])
            w = actualizaVecinos(horizontalesO, w, vel, data[i,:])        
    
            #Diagonal
            diagonalNE = entorno(fila, columna, 1, -1, ra_vecindad, hn, wn)        
    #        print("diagonalNE {}".format(diagonalNE))        
    
            w = actualizaVecinos(diagonalNE, w, vel, data[i,:])
    
    
            diagonalNO = entorno(fila, columna, -1, -1, ra_vecindad, hn, wn)        
    #        print("diagonalNO {}".format(diagonalNO))        
    
            w = actualizaVecinos(diagonalNO, w, vel, data[i,:])
    
    
            diagonalSE = entorno(fila, columna, 1, 1, ra_vecindad, hn, wn)        
    #        print("diagonalSE {}".format(diagonalSE))        
    
            w = actualizaVecinos(diagonalSE, w, vel, data[i,:])
    
            diagonalSO = entorno(fila, columna, -1, 1, ra_vecindad, hn, wn)        
    #        print("diagonalSO {}".format(diagonalSO))        
        
            w = actualizaVecinos(diagonalSO, w, vel, data[i,:])

    return w_activ
        
np.random.seed(190000)
reader = np.genfromtxt("files/XOR_trn.csv", delimiter=',')
data =  reader[:, 0:2] 
w_activ = som(data, 2, 2)    

color = list(["k", "g", "b", "r"])

for i in range(data.shape[0]):
    plt.scatter(data[i, 0], data[i, 1], c = color[int(w_activ[i,0])])
plt.show()