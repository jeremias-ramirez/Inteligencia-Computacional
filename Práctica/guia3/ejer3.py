import numpy as np
from matplotlib import pyplot as plt
import membresia as mem

m = np.array([[-20, -20, -10, -5], [-10, -5, -5, -2], [-5, -2, -2, 0], [-2, 0, 0, 2], [0, 2, 2, 5], [2, 5, 5, 10], [5, 10, 20, 20]])
y = mem.matrizMembresia1E(-7, m, 1)
print(y)
plt.show()

