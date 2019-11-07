# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:37:26 2019

@author: Messi
"""
import sigmoide as sigm
import numpy as np
import initialize_w as init
#posicionados en una capa recibimos el vector de entradas y la
#matriz de pesos. Con estos elementos calculamos las salidas no 
#lineales de cada capa.

#inputV = np.array([1,1,1])
#w = init.initialize_w(inputV,[3, 2, 1])

def salidasy(input_vector, w, funCapas):
        y = []

        y.append(input_vector) #el algoritmo toma la entrada como una salida, se incluye el -1 

        b = 1 #constante de sigmoide
        niveles = len(w)
        
        input_capa = input_vector
        
        for i in range(niveles):
            z = w[i] @ input_capa
            y_aux = np.zeros((np.size(z), 1), np.float)
            
            for j in range(np.size(z)): 
                #y_aux[j] = 1 if z[j] >= 0 else -1
                
                if funCapas[i] == "sigmoid":
                    y_aux[j] = sigm.sigmoide(z[j], b)
                else :
                    y_aux[j] = z[j]

            y_aux = np.insert(y_aux,0,-1,0)
            y.append(y_aux)
            input_capa = y_aux
        
        return y

#saly = salidasy(inputV,w)
#print(saly)
