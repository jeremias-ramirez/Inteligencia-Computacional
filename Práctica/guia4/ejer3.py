import numpy as np
import coloniaHormigas as cH

M = np.genfromtxt("7cities.csv", delimiter=",")

N = 10

result = cH.coloniaHormigas(M, N, maxIter = 100, alpha = 1, beta = 1)
print(result)

