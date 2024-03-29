import numpy as np
from matplotlib import pyplot as plt
import membresia as mem

M = np.array([[-20, -20, -10, -5], [-10, -5, -5, -2], [-5, -2, -2, 0], [-2, 0, 0, 2], [0, 2, 2, 5], [2, 5, 5, 10], [5, 10, 20, 20]])
S = np.array([[-7,-5, -5, -3], [-5, -3, -3, -1], [-3, -1, -1, 0], [-1, 0, 0, 1], [0, 1, 1, 3], [1, 3 , 3, 5], [3, 5, 5, 7]])

x = 0.0
print(mem.matrizMembresia1E(x, M, 1))
print( list( map(  lambda vConj: mem.areaCentr_Trapezoide(1, vConj), S)))

r1 = np.array([1, 2, 3, 4, 5 ,6 , 7])
y = mem.sistemaBorroso(x, r1, M, S, 1) 
print(y)
r2 = np.array([4, 2, 3, 1, 5 ,6 , 7])
y = mem.sistemaBorroso(x, r2, M, S, 1) 
print(y)
