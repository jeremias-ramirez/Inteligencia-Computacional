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
def derLineNoLine(y):
    return list(map(lambda v: 0.5*(1-v)*(1+v),y))

# w: es una lista con las matrices de los pesos, es decir w(0) = matriz de peso de la capa uno
# y : es una lista con los vectores de  salida de las capas, el -1 de w0 debe estar incluido
# yd : vector de salida deseada para calucar el error en la ultma capa, ademas las entradas se considera la primera salida

# vel : velocida de aprendizaje


    
def backpropagation(w,y,yd,vel):
    errorV = yd[:]-y[-1][1:] # el ultimo vector de las lista y no tengo en cuenta el -1
    derL_NL = derLineNoLine(y)
    delta = []

    delta.append(errorV * derL_NL[-1][1:]) # el [1:] se utiliza para no tener en cuenta la salida de y = -1
    # la multiplicacion para dos vectores numpy de las misma dimension se hace elemento a elemento
    wNew = []
    wNew.append(w[-1] + vel * delta[-1] @ y[-2].T)
    for i in range(len(w)-2,-1,-1):
        delta.insert(0,(w[i+1].T @ delta[0]) * derL_NL[i][1:])
        wNew.insert(0,w[i] + vel * (delta[0] @ y[i-1].T))
    return w
    
    
