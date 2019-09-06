import numpy as np
import initialize_w as initW
import salidasy as salY
import backpropagation as bp

def test_server_arquit(data,arquitVec,epoc):
    
    H, W = data.shape
    accurV = np.zeros((k,1))
    porTest = 1/k
    porTrn = round(1 - porTest, 2)
    
    indexs = np.arange(0, H, 1)
    np.random.shuffle(indexs)

    
    for i in range(k):
        tupla = getPartitionsk_fold(data,indexs,porTrn,i)
        dataTrain = data[tupla[1],:]
        H, W = dataTrain.shape
        trn = np.append(-np.ones((len(dataTrain[:,1]),1)),dataTrain[:, 0:W-1],1)
        yd = dataTrain[:, W-1]
        w = np.random.uniform(-0.5,0.5,W)
            
        for j in range (epoc):
            w = train.trainning(trn,yd, w, vel)
            desempeñoV = val.validation(trn,yd, w)
            if desempeñoV > tasa:
                break
        
        dataTest = data[tupla[0],:]
       
        H, W = dataTest.shape
        test = np.append(-np.ones((len(dataTest[:,1]),1)),dataTest[:, 0:W-1],1)
        yd = dataTest[:, W-1]
    
        desempeñoP = val.validation(test,yd, w)
        accurV[i]=desempeñoP       

    plt.plot(range(0,k), accurV)
    plt.show()
 
reader = np.genfromtxt("files/XOR_trn.csv", delimiter=',')

trn = np.append(-np.ones((len(reader[:,1]),1)),reader[:, 0:2],1)
yd = np.expand_dims(reader[:, 2], axis=1)


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
        y = salY.salidasy(inputV, w)
        ye = yd[j]
        accur = (accur + 1  if abs(ye - y[-1][-1]) < 0.3 else accur) 

    accurV[i]=accur/len(trn[:,1])    

print(np.mean(accurV))

