import numpy as np
from matplotlib import pyplot as plt
import membresia as mem


y = mem.defuzzificacion2([0, 0.7, 0.3, 0, 0, 0, 0], np.array([[-7,-5, -5, -3], [-5, -3, -3, -1], [-3, -1, -1, 0], [-1, 0, 0, 1], [0, 1, 1, 3], [1, 3 , 3, 5], [3, 5, 5, 7]]), 1)

m2 = np.array([[-10, 1], [-5, 1], [0, 1], [5, 1], [10, 1]])
y = mem.defuzzificacion2([0, 0.7, 0.3, 0, 0, 0, 0],m2, 2)

print(y)


