import numpy as np
import k_medias as km
from matplotlib import pyplot as plt
#np.random.seed(124394140)
reader = np.genfromtxt("files/XOR_tst.csv", delimiter=',')

data =  reader[0:10, 0:2] 
H, W = data.shape

indexs = np.arange(0, H, 1)
np.random.shuffle(indexs)
k = 4
grupos = km.k_media_batch(data,indexs, k)
color = list(["k", "g", "b", "r"])

for i in range(k):
    plt.scatter(data[indexs[grupos == i], 0], data[indexs[grupos == i], 1], c = color[i])
plt.show()
