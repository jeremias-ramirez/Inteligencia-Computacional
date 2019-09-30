import numpy as np
import math
#distancia promedio, entre las distancias promedios de los centroides con sus datos asociados

def distCent(data, indexs, gruposCent, centroides, k):
    tamGrupoCent = lambda i : indexs[ gruposCent == i].shape[0]
    sumCentroide = lambda i : np.sum (np.linalg.norm( (data[ indexs[ gruposCent == i], :] - centroides[i]), axis = 1, ord = 2)) / tamGrupoCent(i)

    return sum([sumCentroide(indexCent) for indexCent in range(k)]) / k

#data = datos
#index: vector numpy con la indeces mezclados
#k: cantidad de cluster
#tol: tolerancia de corte para el promedio de distancias entre los centroides y sus datos asociados

def k_media_batch(data, indexs, k, tol=0.1):
    
    H, W = data.shape
    #funcion para elegir centroides al azar
    getCentroides = lambda : [ data[np.random.randint(0,H), :] for i in range(k)]
    
    gruposCentroides = np.zeros((H))
    distancias = np.zeros((H,k)) 
    iterMax = 200
    distancia = tol+1
    # el algoritmo corta por una tolerancia
    # a partir de unos centroides elegidos aleatoriamente itera para armar los cluster(grupos)
    # luego si finaliza porque los ya no hay cambio en los grupos, o el fin de iteraciones 
    # calula la distancia promedio, para verificar si finalizar con esos grupos, y centroides o
    # elegir nuevamente otro centroides aleatorios, y luego hacer la iteracion nuevamente
    while distancia > tol:

        centroides = getCentroides()
        
        for n in range(iterMax):
            
            for i, centroide in enumerate(centroides):
                distancias[:, i] = np.linalg.norm (data[indexs[:],:] - centroide, ord = 2, axis = 1)

            gruposCentroidesNew = np.argmin(distancias, axis = 1)
            
            if (gruposCentroidesNew == gruposCentroides).all():
                break

            gruposCentroides = gruposCentroidesNew
            centroides = [ np.mean( data[ indexs[ gruposCentroides == indexCent], :], axis = 0) for indexCent in range(k)]
        
        # si hay algun centroide el cual no tiene asociado ningun dato, busca otros centroides aleatorios 
        if  any(list([indexs[gruposCentroidesNew == i].shape[0] == 0 for i in range(k)])):
            continue
        
        distancia = distCent(data, indexs, gruposCentroides, centroides, k)

    return  gruposCentroides 

