import numpy as np
import k_medias as km
import multiprocessing as mp
from matplotlib import pyplot as plt

np.random.seed(124394140)

reader = np.genfromtxt("files/irisbin.csv", delimiter=',')

data =  reader[:, 0:4] 
y = np.expand_dims(reader[:, 4:], axis = 1)


H, W = data.shape

indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)

minK = 4
maxK = 30

distPromV = np.ones((maxK-minK,1))

pool = mp.Pool(processes = maxK-minK)
results = [pool.apply(km.k_medias_tol, (data,indexs, k, 0.05, 200, 200, )) for k in range(minK, maxK)]


for k in range(minK, maxK):
    distPromV[k-minK] = results[k-minK][1]

plt.plot(np.arange(minK, maxK), distPromV)
plt.scatter(np.arange(minK, maxK), distPromV)

plt.xticks(np.arange(minK, maxK))
plt.show()

#transfGauss = km.gauss_k_medias(data, centroides, k)
#outputGauss = np.append(transfGauss, y, axis = 1)
#
#np.savetxt("files/iris_k_medias_" + str(k) + ".csv", outputGauss, delimiter = ",")


