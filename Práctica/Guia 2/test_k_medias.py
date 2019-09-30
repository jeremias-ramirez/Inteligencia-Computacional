import numpy as np
import k_medias as km
from matplotlib import pyplot as plt
import multiprocessing as mp

np.random.seed(124394140)
reader = np.genfromtxt("files/XOR_tst.csv", delimiter=',')

data =  reader[:, 0:2] 
y = np.expand_dims(reader[:, 2], axis = 1)


H, W = data.shape

indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)

minK = 1
maxK = 15

pool = mp.Pool(processes = maxK-minK)

distPromV = np.ones((maxK,1))

results = [pool.apply_async(km.k_medias_tol, (data,indexs, k, 0.05, 200, 200, )) for k in range(1, maxK+1)]
output = [p.get() for p in results]

[print(result[1]) for result in output]

#plt.plot(np.arange(1, maxK+1), distPromV)

#k = 4
#centroides, distProm, grupos = km.k_medias_tol(data,indexs, k)
#for i in range(k):
#    plt.scatter(data[indexs[grupos == i], 0], data[indexs[grupos == i], 1])
#    plt.scatter(centroides[i][0], centroides[i][1], marker="^")

plt.show()

#transfGauss = km.gauss_k_medias(data, centroides, k)
#outputGauss = np.append(transfGauss, y, axis = 1)
#np.savetxt("files/test_k_medias.csv", outputGauss, delimiter = ",")


