# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:45:16 2019

@author: Shirli
"""
import numpy as np
from matplotlib import pyplot as plt


## determina las neuronas conectadas entre si segun el radio de vecindad.. devuelve un vector de indices que tiene
## la fila y columna de la neurona vecina
def entorno(fila, columna, paso_horizontal, paso_vertical, ra_vecindad, hn, wn):
    flag = True
    filaI = fila
    columnaI = columna
    indices = list()
    while flag:
        filaI = filaI + paso_vertical
        columnaI = columnaI + paso_horizontal 
        if filaI >= hn or columnaI >= wn:
            break
        if filaI < 0 or columnaI < 0 :
            break
        distancia = np.linalg.norm(np.array([filaI - fila, columnaI - columna]))
        if distancia > ra_vecindad:
            break
        else:
            indices.append((filaI,columnaI)) 
    return indices

## Adapto los pesos de todas las neuronas vecinas que son afectadas por la neurona ganadora
def actualizaVecinos(listaVecinos, w, vel, datai):
    for i in range(len(listaVecinos)):
        columna = listaVecinos[i][1]
        fila = listaVecinos[i][0]
        w[fila,columna] = w[fila,columna] + vel * (datai-w[fila,columna])
    return w

def som(data, hn, wn):
    ra_vecindad = 0
    H, W = data.shape
    w = np.zeros((hn,wn,W))
    indexsH = np.arange(0, H, 1)
    indexsW = np.arange(0, W, 1)
    np.random.shuffle(indexsH)
    np.random.shuffle(indexsW)   
    vel = 0.01
    
    w_activ = np.zeros((H,1))
    epoc = 500
    
    ## Inicializo los pesos al azar
    for i in range(hn):
        for j in range(wn):
#            w[i,j,:] = data[indexsH[i]]
             w[i,j,:] = data[np.random.randint(0, H)]
#    print(w)
#    print(w.reshape((hn*wn, W))
    for ep in range(epoc):
        ## vario el radio de vecindad segun las epocas
        if ep < 15:
            ra_vecindad = 2 #muy poquitas neuronas con XOR sino deberia ser 2
        elif ep < 50:
            ra_vecindad = 1
        else:
            ra_vecindad = 0
#        print("epoca {} ra {}".format(ep,ra_vecindad))
        ## Para cada datos determino la ganadora y sus vecinos 
        for i in range(H):
    #        print(data[i,:])
            distancias = np.linalg.norm((data[i, :] - w.reshape((hn*wn, W) )), axis = 1,ord=2)
    #        print(distancias)
        
            ind = np.argmin(distancias, axis = 0)
#            print(ind)
            columna = ind%wn
            fila = int(ind/wn)
            if ra_vecindad == 0:
                w_activ[i]= int(ind)
            # actual
    #        print("fila, columna {} {}".format(fila, columna))
            w[fila,columna] = w[fila,columna] + vel * (data[i,:]-w[fila,columna])
           
            #vertical
            verticalesS = entorno(fila, columna, 0, 1, ra_vecindad, hn, wn)        
    #        print("verticalesS {}".format(verticalesS))
            verticalesN = entorno(fila, columna, 0, -1, ra_vecindad, hn, wn)        
    #        print("verticalesN {}".format(verticalesN))
    
    #        print("W viejo {}, verticalesS {}, verticalesN {}".format(w, verticalesS, verticalesN))
            
            w = actualizaVecinos(verticalesS, w, vel, data[i,:])
            w = actualizaVecinos(verticalesN, w, vel, data[i,:])        
    
    #        print("W actualizado {}".format(w))
            #horizontal
            horizontalesE = entorno(fila, columna, 1, 0, ra_vecindad, hn, wn)        
    #        print("horizontalesE {}".format(horizontalesE))
            horizontalesO = entorno(fila, columna, -1, 0, ra_vecindad, hn, wn)        
    #        print("horizontalesO {}".format(horizontalesO))
            w = actualizaVecinos(horizontalesE, w, vel, data[i,:])
            w = actualizaVecinos(horizontalesO, w, vel, data[i,:])        
    
            #Diagonal
            diagonalNE = entorno(fila, columna, 1, -1, ra_vecindad, hn, wn)        
    #        print("diagonalNE {}".format(diagonalNE))        
    
            w = actualizaVecinos(diagonalNE, w, vel, data[i,:])
    
    
            diagonalNO = entorno(fila, columna, -1, -1, ra_vecindad, hn, wn)        
    #        print("diagonalNO {}".format(diagonalNO))        
    
            w = actualizaVecinos(diagonalNO, w, vel, data[i,:])
    
    
            diagonalSE = entorno(fila, columna, 1, 1, ra_vecindad, hn, wn)        
    #        print("diagonalSE {}".format(diagonalSE))        
    
            w = actualizaVecinos(diagonalSE, w, vel, data[i,:])
    
            diagonalSO = entorno(fila, columna, -1, 1, ra_vecindad, hn, wn)        
    #        print("diagonalSO {}".format(diagonalSO))        
        
            w = actualizaVecinos(diagonalSO, w, vel, data[i,:])

    return w_activ
        
#np.random.seed(190000)
reader = np.genfromtxt("files/clouds.csv", delimiter=',')
data =  reader[:, 0:2] 
result = reader[:, 2]
w_activ = som(data, 6, 6)    

#color1 = list(["b", "r", "k", "g", ""])
color = ['black',    'silver',    'red',        'gold',     'orange',   'salmon', 'green',      
         'blue',   'blueviolet',  'purple',    'fuchsia',   'yellow',
         'mediumspringgreen', 'lightseagreen',    'sienna',     'moccasin', 'chartreuse',
          'darkcyan', 'royalblue', 'pink',     'tan',  'olivedrab',  'tomato','turquoise', 
          'black',    'silver',    'red',        'gold',     'orange',   'salmon', 'green',      
         'blue',   'blueviolet',  'purple',    'fuchsia',   'yellow',
         'mediumspringgreen', 'lightseagreen',    'sienna',     'moccasin', 'chartreuse',]
color2 = list(["k", "g"])

#clase1 = 0
#clase2 = 0
#
#clase1_1 = 0
#clase1_2 = 0
#
#clase2_1 = 0
#clase2_2 = 0

v_clases = np.zeros((36,1))
clase1 = np.zeros((v_clases.shape[0], 2))
#clase2 = np.zeros((v_clases.shape[0], 2))

for i in range(data.shape[0]):

#    if int(w_activ[i,0])> 22:
#        col += 1 
#    else:
#        col = int(w_activ[i,0])
    col = int(w_activ[i,0])
    plt.scatter(data[i, 0], data[i, 1], c = color[col])
#    if w_activ[i, 0] == 0:
#        clase1 += 1
#        if result[i] == 0: 
#            clase1_1 += 1
#        else:
#            clase1_2 += 1
#    else:
#        clase2 += 1
#        if result[i] == 0: 
#            clase2_1 += 1
#        else:
#            clase2_2 += 1
    j = int(w_activ[i,0])
    v_clases[j] += 1 
    if result[i] == 0: 
        clase1[j][0] += 1
    else:
        clase1[j][1] += 1
plt.show()

for i in range(data.shape[0]):

    plt.scatter(data[i, 0], data[i, 1], c = color2[int(result[i])])
    
plt.show()    

#print("Clase 1: {}".format(clase1))
#print("Clase 2: {}".format(clase2))

for i in range(clase1.shape[0]):    
    print("Clase {} que son 0: {}".format(i,clase1[i][0]))
    print("Clase {} que son 1: {}".format(i, clase1[i][1]))
    
m_confusion = np.zeros((4,1))

for i in range(clase1.shape[0]):
    if clase1[i][0] > clase1[i][1]:
        print("La clase {} representa los 0 con el color {}".format(i, color[i]))
    else:
        print("La clase {} representa los 1 con el color {}".format(i, color[i]))
        

    m_confusion[0] = (m_confusion[0] + clase1[i][0]) if clase1[i][0] > clase1[i][1] else (m_confusion[0])
    m_confusion[1] = (m_confusion[1] + clase1[i][1]) if clase1[i][0] > clase1[i][1] else (m_confusion[1])
    m_confusion[2] = (m_confusion[2] + clase1[i][0]) if clase1[i][0] < clase1[i][1] else (m_confusion[2])
    m_confusion[3] = (m_confusion[3] + clase1[i][1]) if clase1[i][0] < clase1[i][1] else (m_confusion[3])

print("verdadero - verdadero {} - {}%".format(m_confusion[0], (m_confusion[0]/(m_confusion[0]+m_confusion[1]))*100))
print("verdadero - falso {} - {}%".format(m_confusion[1], (m_confusion[1]/(m_confusion[0]+m_confusion[1]))*100))
print("falso - verdadero {} - {}%".format(m_confusion[2], (m_confusion[2]/(m_confusion[2]+m_confusion[3]))*100))
print("falso - falso {}- {}%".format(m_confusion[3], (m_confusion[3]/(m_confusion[2]+m_confusion[3]))*100))



#print("Clase 2 que son 0: {}".format(clase2_1))
#print("Clase 2 que son 1: {}".format(clase2_2))

