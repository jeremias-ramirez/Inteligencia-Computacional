import numpy as np
import algoritmosGeneticos as ag

v = np.array([10, 10, 10, 10, 10, 50])

result = ag.getRuleta(v)
print(result)

poblacion = np.array([[-1, 0, 0, 1], [0, 1, 1, 3], [1, 3 , 3, 5], [3, 5, 5, 7], [1, 2,3, 4], [5, 6, 7, 8], [2, 4, 5, 5], [3, 4, 4, 8]])

def fitness(v):
    return sum(v)

print(list(map(fitness, poblacion)))

padres = ag.seleccion(poblacion,fitness, 0.5)

print(list(map(fitness, padres)))
