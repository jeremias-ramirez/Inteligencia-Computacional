import numpy as np
import k_medias as km

reader = np.genfromtxt("files/merval.csv", delimiter=',')

data = np.array([ reader[i * 5 : (i+1) * 5] for i in range(int(reader.shape[0] / 5)) ])

inData = data[:, 0:4]
yd = np.expand_dims(data[:,4], axis = 1)

H, W = inData.shape

indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)

k = 14

centroides, distancia, _ = km.k_medias_tol(inData,indexs, k, 0.05, 500)

transfGauss = km.gauss_k_medias(inData, centroides, k)

outputGauss = np.append(transfGauss, yd, axis = 1)

np.savetxt("files/merval_k_medias_" + str(k) + ".csv", outputGauss, delimiter = ",")


