import numpy as np
import partitions as pt

H = 20
indexs = np.arange(0,H)
k = 4
cantPar = int(H/k)

for i in range(cantPar):
    [test, train] = pt.getPartitions_leave_k_out(indexs, k, i)
    print("test: {}, train: {}".format(test,train))

