import numpy as np
from matplotlib import pyplot as plt
import membresia as mem

s1 = np.array([[-7,-5, -5, -3], [-5, -3, -3, -1], [-3, -1, -1, 0], [-1, 0, 0, 1], [0, 1, 1, 3], [1, 3 , 3, 5], [3, 5, 5, 7]])
a = np.array([0, 0.7, 0.3, 0, 0, 0, 0])
y1 = mem.defuzzificacion(a, s1, 1)
print(y1)

s2 = np.array([[-10, 1], [-5, 1], [0, 1], [5, 1], [10, 1]])
y2 = mem.defuzzificacion(a,s2, 2)

print(y2)


