import numpy as np
import k_medias as km

import sys
sys.path.append("../")

import trainning as trnD
from matplotlib import pyplot as plt


merval = np.genfromtxt("files/merval.csv", delimiter=',')
data = np.genfromtxt("files/merval_k_medias_20.csv", delimiter=',')

prediccion = np.concatenate((merval[0:5], np.sum(data, axis = 1)), axis = 0)

figMerval = plt.figure(1)

axMerval = figMerval.add_subplot(111)

axMerval.set_title("Curva Merval ")

axMerval.set_ylabel(" Valor del Indice Merval ")


dias = np.arange(0, merval.shape[0])
dias_6 = np.arange(5,merval.shape[0],6)

axMerval.set_xlabel(" Tiempo(Dias) ")
#axMerval.set_xticks(dias_6)
#, dias_6, np.sum(data, axis = 1), "k." 
axMerval.plot(dias, merval, dias_6, np.sum(data, axis = 1))
axMerval.legend(("Correcto", "Prediccion"), loc = "upper left")

plt.show()


