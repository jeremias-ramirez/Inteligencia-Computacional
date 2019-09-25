import numpy as np
import k_medias as km
from matplotlib import pyplot as plt

reader = np.genfromtxt("files/XOR_tst.csv", delimiter=',')

data =  reader[:, 0:2] 

grupos = km.k_media_batch(data, 4)
color = list(["k", "g", "b", "r"])

for i, grupo in enumerate(grupos):
    plt.scatter(data[grupo, 0], data[grupo, 1], c = color[i])
plt.show()
