
import numpy as np
from matplotlib import pyplot as plt
import membresia as mem


M1 = np.array([[-20, -20, -10, -5], [-10, -5, -5, -2], [-5, -2, -2, 0], [-2, 0, 0, 2], [0, 2, 2, 5], [2, 5, 5, 10], [5, 10, 20, 20]])

M2 = np.array([[-20, -20, -10, -5], [-10, -5, -4, -2], [-4, -2, -1, 0], [-1, 0, 0, 1], [0, 1, 2, 4], [2, 4, 5, 10], [5, 10, 20, 20]])

S1 =np.array([[-7,-5, -5, -3], [-5, -3, -3, -1], [-3, -1, -1, 0], [-1, 0, 0, 1], [0, 1, 1, 3], [1, 3 , 3, 5], [3, 5, 5, 7]])

S2 =np.array([[-7,-5, -5, -4], [-5, -4, -4, -3], [-4, -3, -3, 0], [-3, 0, 0, 3], [0, 3, 3, 4], [3, 4 ,4, 5], [4, 5, 5, 7]])

def contrT_borroso(e, M, S):
    gA = mem.matrizMembresia1E(e, M, 1)
    return mem.defuzzificacion(gA, S, 1)



def contrTemperatura(tempD, toIni, M, S):

    to = np.zeros((tempD.shape[0]))
    q = np.zeros((tempD.shape[0]))

    e = tempD[0] - toIni
    q[0] = contrT_borroso(e, M, S)
    
    g = 40 / 41
    a = 40 / 41
    ti = 15
    toN = lambda toAnt, q: ti + g * q + a* (toAnt - ti)
    
    to[0]= toN(toIni, q[0])
    for i in range(tempD.shape[0] - 1):
        e = tempD[i] - to[i]
        q[i+1] = contrT_borroso(e, M, S)
        to[i+1]= toN(to[i], q[i+1])

    return to, q


temp = np.ones((200))
temp[0:30] = 15 * temp[0:30]
temp[30:] = 25 * temp[30:]
toIni = 15

to_M1S1, q1 = contrTemperatura(temp, toIni, M1, S1)
to_M1S2, q2 = contrTemperatura(temp, toIni, M1, S2)

to_M2S1, q3 = contrTemperatura(temp, toIni, M2, S1)
to_M2S2, q4 = contrTemperatura(temp, toIni, M2, S2)

tiempo = np.arange(0,200)

fig = plt.figure(1)

ax1 = fig.add_subplot(221)
ax1.set_title("Entrada M1 Salida S1")
ax1.plot(tiempo, temp, tiempo, to_M1S1, tiempo, q1)
ax1.legend(("Temp. Deseada", "Temp. Obtenida", "Incremento q"), loc = "center right")

ax2 = fig.add_subplot(222)
ax2.set_title("Entrada M1 Salida S2")
ax2.plot(tiempo, temp, tiempo, to_M1S2, tiempo, q2)
ax2.legend(("Temp. Deseada", "Temp. Obtenida", "Incremento q"), loc = "center right")

ax3 = fig.add_subplot(223)
ax3.set_title("Entrada M2 Salida S1")
ax3.plot(tiempo, temp, tiempo, to_M2S1, tiempo, q3)
ax3.legend(("Temp. Deseada", "Temp. Obtenida", "Incremento q"), loc = "center right")

ax4 = fig.add_subplot(224)
ax4.set_title("Entrada M2 Salida S2")
ax4.plot(tiempo, temp, tiempo, to_M2S2, tiempo, q4)
ax4.legend(("Temp. Deseada", "Temp. Obtenida", "Incremento q"), loc = "center right")


plt.show()


