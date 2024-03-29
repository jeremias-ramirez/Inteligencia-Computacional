# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:48:10 2019

@author: Messi
"""

import numpy as np
import initialize_w as init

#funcion que obtiene la derivada de la funcion no lineal con la lineal
# y es una lista con los vectore de salida de las capas
# no importa que este el -1 agregado, ya que la salida no se utiliza 
def derLineNoLine(y, funCapas):
    def getDer(index):
        if funCapas[index] == "sigmoid":
            return  0.5*(1-y[index])*(1+y[index])
        else :
            return np.ones_like(y[index])
    return [ getDer(i) for i in range(len(funCapas)) ]

# w: es una lista con las matrices de los pesos, es decir w(0) = matriz de peso de la capa uno
# y : es una lista con los vectores de  salida de las capas, el -1 de w0 debe estar incluido
# yd : vector de salida deseada para calcular el error en la ultima capa, ademas las entradas se considera la primera salida

# vel : velocida de aprendizaje


    
def backpropagation(w, y, yd, vel):
    
    derL_NL = derLineNoLine(y[1:,:], funCapas)
    errorV = yd[:] - y[-1][1:]  # el ultimo vector de las lista y no tengo en cuenta el -1
    
    delta = errorV * derL_NL[-1][1:] # el [1:] se utiliza para no tener en cuenta la salida de y = -1
    # la multiplicacion para dos vectores numpy de las misma dimension se hace elemento a elemento
    
    wNew = []
    wNew.append(w[-1] + vel * (delta @ y[-2].T))
    
    wLen = len(w)
    yLen = wLen + 1
    
    # el tamaño de y debe ser wLen + 1 porque y considera a la entrada como una salida y ademas tienen la salida de la
    # ultima capa
    for i in range(0,wLen-1):

        delta =  (w[wLen - 1 - i][:,1:].T @ delta) * derL_NL[yLen - 1 - i][1:]
        wNew.insert(0, w[wLen - 2 - i] + vel * (delta @ y[yLen - 3 - i].T))

    return wNew

def backpropagation_momento(w, y, yd, vel, velM, funCapas, deltaW = None):

    derL_NL = derLineNoLine(y[1:], funCapas)
    
    errorV = yd[:].T - y[-1][1:] # el ultimo vector de las lista y no tengo en cuenta el -1
    delta = errorV * derL_NL[-1][1:] # el [1:] se utiliza para no tener en cuenta la salida de y = -1
    
    # la multiplicacion para dos vectores numpy de las misma dimension se hace elemento a elemento
    if deltaW == None:
        deltaW = [np.zeros_like(we) for we in w]
    
    deltaW[-1] = vel * (delta @ y[-2].T) + velM * deltaW[-1]
    wNew = []
    wNew.append(w[-1] +  deltaW[-1])
    
    wLen = len(w)
    yLen = wLen + 1
    

    # el tamaño de y debe ser wLen + 1 porque y considera a la entrada como una salida y ademas tienen la salida de la
    # ultima capa
    for i in range(0,wLen-1):

        delta =  (w[wLen - 1 - i][:,1:].T @ delta) * derL_NL[wLen - 2 - i][1:]
        deltaW[wLen - 2 - i] = vel * (delta @ y[yLen - 3 - i].T) + velM * deltaW[wLen - 2 - i] 
        wNew.insert(0, w[wLen - 2 - i] + deltaW[wLen - 2 - i])

    return wNew, deltaW
    
    
