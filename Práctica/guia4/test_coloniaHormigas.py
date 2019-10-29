import numpy as np
import coloniaHormigas as cH

M = np.genfromtxt("7cities.csv", delimter=",")

M = np.array([[1, 1 , 20, 20], [1, 1, 20, 20], [1, 1, 20, 20], [1, 1, 20, 20]])
N = 3

result = cH.coloniaHormigas(M, N, maxIter = 100, alpha = 1, beta = 1)
print(result)

