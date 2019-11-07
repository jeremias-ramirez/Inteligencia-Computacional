import numpy as np
import k_medias as km

reader = np.genfromtxt("files/merval.csv", delimiter=',')
#armar la base de datos
data = np.array([ reader[i * 6 : (i+1) * 6] for i in range(int(reader.shape[0] / 6)) ])
inData = data[:, 0:5]
yd = np.expand_dims(data[:,5], axis = 1)

H, W = inData.shape
indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)

k = 20

centroides, distancia, grupoCentroides = km.k_medias_tol(inData,indexs, k, 0.05, 500)

#obtener una varianza promedio
varianzas = np.array([np.var(data[indexs[grupoCentroides == i], : ], axis = 0) for i in range(k)])
var = np.mean( np.mean(varianzas, axis = 1))

#transformacion
transfGauss = km.gauss_k_medias(inData, centroides, k, int(var))

outputGauss = np.append(transfGauss, yd, axis = 1)

np.savetxt("files/merval_k_medias_" + str(k) + ".csv", outputGauss, delimiter = ",")


