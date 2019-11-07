import numpy as np
import k_medias as km
from matplotlib import pyplot as plt


reader = np.genfromtxt("files/merval.csv", delimiter=',')
#armar al base de datos
data = np.array([ reader[i * 6 : (i+1) * 6] for i in range(int(reader.shape[0] / 6)) ])
inData = data[:, 0:5]
yd = np.expand_dims(data[:,5], axis = 1)

H = inData.shape[0]
indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)

minK = 15
maxK = 35

distPromV = np.ones((maxK-minK,1))

results = [km.k_medias_tol(inData, indexs, k, 0.05, 400, 200) for k in range(minK, maxK)]

for k in range(minK, maxK):
    distPromV[k-minK] = results[k-minK][1]


fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(np.arange(minK, maxK), distPromV)
ax.scatter(np.arange(minK, maxK), distPromV)

ax.set_title("Curva k_medias Merval")
ax.set_xticks(np.arange(minK, maxK))
ax.set_xlabel(" K ")
ax.set_ylabel("Distancia Promedio de Centroides")

plt.show()

