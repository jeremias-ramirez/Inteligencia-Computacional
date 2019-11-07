import numpy as np
import k_medias as km
from matplotlib import pyplot as plt


reader = np.genfromtxt("files/irisbin.csv", delimiter=',')

data =  reader[:, 0:4] 
y = np.expand_dims(reader[:, 4:], axis = 1)

H, W = data.shape

indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)

minK = 4
maxK = 27

distPromV = np.ones((maxK-minK,1))

results = [km.k_medias_tol(data,indexs, k, 0.05, 200, 200) for k in range(minK, maxK)]


for k in range(minK, maxK):
    distPromV[k-minK] = results[k-minK][1]


fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(np.arange(minK, maxK), distPromV)
ax.scatter(np.arange(minK, maxK), distPromV)

ax.set_title("Curva k_medias Iris")
ax.set_xticks(np.arange(minK, maxK))
ax.set_xlabel(" K ")
ax.set_ylabel("Distancia Promedio de Centroides")

plt.show()

