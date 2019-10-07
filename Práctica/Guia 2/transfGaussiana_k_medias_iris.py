import numpy as np
import k_medias as km


reader = np.genfromtxt("files/irisbin.csv", delimiter=',')

data =  reader[:, 0:4] 
y = reader[:, 4:]

H, W = data.shape
indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)

k = 16
centroides, distancia, _ = km.k_medias_tol(data,indexs, k, 0.05, 500)

transfGauss = km.gauss_k_medias(data, centroides, k)

outputGauss = np.append(transfGauss, y, axis = 1)

#np.savetxt("files/iris_k_medias_" + str(k) + ".csv", outputGauss, delimiter = ",")


