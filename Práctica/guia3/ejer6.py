import numpy as np
from matplotlib import pyplot as plt
import membresia as mem


M = np.array([[-20, -20, -10, -5], [-10, -5, -5, -2], [-5, -2, -2, 0], [-2, 0, 0, 2], [0, 2, 2, 5], [2, 5, 5, 10], [5, 10, 20, 20]])
S = np.array([[-7,-5, -5, -3], [-5, -3, -3, -1], [-3, -1, -1, 0], [-1, 0, 0, 1], [0, 1, 1, 3], [1, 3 , 3, 5], [3, 5, 5, 7]])
r = np.array([1, 2, 3, 4, 5 ,6 , 7])
x = np.arange(-20, 20, 0.5)
y = mem.sistemaBorrosoMap(x, r, M, S, 1) 

r2 = np.array([7, 2, 6, 3, 1 ,4 , 5])
y2 = mem.sistemaBorrosoMap(x, r2, M, S, 1) 
plt.figure(1)
plt.plot(x, y, x, y2)

M2 = np.array([[-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1]])
S2 = np.array([[-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1]])
r2 = np.array([1, 2, 3, 4, 5])
x = np.arange(-2, 2, 0.5)
y2 = mem.sistemaBorrosoMap(x, r2, M2, S2, 2) 

plt.figure(2)
plt.plot(x, y2)

plt.show()
