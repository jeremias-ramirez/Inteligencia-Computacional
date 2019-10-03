import numpy as np
from matplotlib import pyplot as plt
import membresia as mem

x = np.arange(-4,4,0.1)

#y1 = list([membresia(xe, [-4,-3,-1,2], 1) for xe in x])
#y2 = list([membresia(xe, [1,1], 2) for xe in x])

#plt.plot(x, y1, x, y2)

m = np.array([[-20, -20, -10, -5], [-10, -5, -5, -2], [-5, -2, -2, 0], [-2, 0, 0, 2], [0, 2, 2, 5], [2, 5, 5, 10], [5, 10, 20, 20]])

m2 = np.array([[-10, 1], [-5, 1], [0, 1], [5, 1], [10, 1]])
x = np.arange(-20, 20)
y = mem.matrizMembresia(x, m, 1)

x2 = np.arange(-12, 12, 0.05)

y = mem.matrizMembresia(x, m, 1)
y2 = mem.matrizMembresia(x2, m2, 2)

list(map(lambda ye : plt.plot(x, ye), y))

#list(map(lambda ye : plt.plot(x2, ye), y2))


plt.show()

