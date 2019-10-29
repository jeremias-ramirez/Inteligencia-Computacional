import numpy as np
from math import inf

def getIndexRuleta(ruleta, pGiro, index = 0):
    if ruleta[index] > pGiro:
        return index
    else:
        return getIndexRuleta(ruleta, pGiro, index + 1)

#M : matriz de los caminos, cada elemento ij indica la distancia entre el elemento i y j
#N : cantidad de hormigas
def coloniaHormigas(M, N, valorDesado = 10, maxIter = 100,  alpha = 0.01, beta = 0.001, rho = 0.01):
    sigma = np.ones_like(M) #incio de las feromonas
    destinos = set(range(M.shape[0])) # conjunto de todos los caminos posibles
    origen = 6
    wayMinH = list()
    distMinH = inf
    valorMin = valorDesado + 1
    it = 0

    def getWayH(wayH):
        wayPosseH  = list( destinos.difference( set( wayH))) # se obtiene los seguiente posibles nodos que puede visitar
        if not wayPosseH : #forma de preguntar si es una lista vacia
            return wayH
        else:
            #funcion de calculo de probabilida del camino de la hormiga, el nodo en que se encuetra , es el ultimo de la lista
            sumatoria = np.power(sigma[wayH[-1], wayPosseH ] , alpha) @ np.power( (1 / M[wayH[-1], wayPosseH]), beta)
            probabilidades =np.array( [np.power(sigma[wayH[-1], i], alpha) * np.power(1 / M[wayH[-1], i], beta) for i in wayPosseH ] / sumatoria )
            
            #armo la ruleta donde se acumula las probabilidades, 
            ruleta = [ np.sum(probabilidades[0:i]) for i in range(1, probabilidades.shape[0])]
            ruleta.append(1.0)
            
            #obtengo la posicion en la lista del elemento y esto lo agrego al principio de la lista 
            index = getIndexRuleta(ruleta, np.random.random()) #obtengo un indice de los elementos, aleatoriamente
            wayH.append(wayPosseH[index])
            return getWayH(wayH)


    while maxIter > it and valorDesado < valorMin:
        it+=1
        wayHs = [np.array (getWayH([origen])) for i in range(N)]
        
        def getMatrixDist(way):
            matriz_unos = np.zeros_like(M) #la matriz siguiente va a tener 1 por donde los nodos paso
            
            for i in range(way.shape[0]-1):
                matriz_unos[way[i], way[i+1]] = 1
            matriz_unos[way[-1], way[0]] = 1 # valor de la vuelta
            
            matriz_dist = matriz_unos * M 

            return matriz_unos, np.sum(matriz_dist)

        matrix1s_Dist = np.array( list( map( getMatrixDist, wayHs)))
        matrixInc = np.sum([matrix1/dist for matrix1, dist in matrix1s_Dist], axis = 0)
        sigma = sigma * (1 - rho) + matrixInc
        
        indexMin = np.argmin(matrix1s_Dist[:,1])
        if matrix1s_Dist[indexMin, 1] < distMinH:
            distMinH = matrix1s_Dist[indexMin, 1]
            wayMinH = wayHs[indexMin]
        valorMin = matrix1s_Dist[indexMin, 1]

    wayMinH = np.append(wayMinH, origen)
    return list(wayMinH+1), distMinH

