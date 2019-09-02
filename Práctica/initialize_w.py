# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 00:32:30 2019

@author: Messi
"""
import numpy as np
import initialize_w as init

#el vector multicapaV contiene en cada elemento la cantidad de neuronas 
#de la capa que corresponde a su posicion

#column es la cant de entradas que tiene la red neuronal incluyendo x0

#esta funcion devuelve una lista o vector de matrices de pesos aleatorios
#el primer indice corresponde al nivel de capa en el que estamos,
#y el segundo indice devuelve el vector de pesos que le corresponde
#a la neurona con dicho indice

#input_vector = np.array([1, 1, 1])
#multicapaV = np.array([3, 2, 1])

def initialize_w(input_vector, multicapaV):
    niveles = np.size(multicapaV)
    nivel0 = np.size(input_vector)
    matrix = []
    matrix_aux = []
    
    f = multicapaV[0]
    matrix_aux = np.random.random((f,nivel0))
    matrix_aux = matrix_aux - 0.5
    matrix.append(matrix_aux)
    
    for i in range(niveles-1): 
         i = i + 1
         f = multicapaV[i]
         matrix_aux = np.random.random((f,multicapaV[i-1] + 1))
         matrix_aux = matrix_aux - 0.5
         matrix.append(matrix_aux)
         #np.array(matrix)
    return matrix
#w = init.initialize_w(input_vector, multicapaV)
