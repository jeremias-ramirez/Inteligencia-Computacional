import numpy as np
import initialize_w as initW
import salidasy as salY
import backpropagation as bp

reader = np.genfromtxt("files/XOR_trn.csv", delimiter=',')

trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
yd = np.expand_dims(reader[:, 2], axis=1)

# inicializo aleatoriamente los pesos W - vector de entradas y vector con la cantidad de neuronas
w = initW.initialize_w( np.ones((len(trn[0,:]),1)), np.array([2,1], np.int ))


vel = 0.1
epoc = 10

accurV = np.zeros((epoc,1))
wV = np.zeros((epoc,3))

errorV = np.zeros((len(trn[:,1]),epoc))
for i in range(epoc):
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis=1)
        y = salY.salidasy(inputV,w)
        w = bp.backpropagation(w,y,yd[j],vel)
    accur = 0
    for j in range(len(trn[:,0])):
        inputV = np.expand_dims(trn[j,:], axis=1)
        y = salY.salidasy(inputV,w)
        ys = 1 if y[-1][-1] >= 0 else -1
        error=yd[j] - ys
        accur = (accur + 1 if error == 0 else accur)
    accurV[i]=accur/len(trn[:,1])    

for i in accurV:
    print(i)

print(np.mean(accurV))

#plt.scatter(trn[:,1],trn[:,2])
#plt.plot(trn[:,1],-trn[:,1]*w[1]/w[2]+w[0]/w[2], 'g')
#plt.show()

#reader = np.genfromtxt("or_tst.csv", delimiter=',')
#trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
#result = reader[:, 2]
#accur = 0
#for j in range(len(trn[:,0])):
#    z = sum(trn[j,:] * w[:])
#    y = 1 if z >= 0 else -1
#    error=result[j] - y
#    accur = (accur + 1 if error == 0 else accur)
#
#print(accur/len(trn[:,1]))    
#
#
