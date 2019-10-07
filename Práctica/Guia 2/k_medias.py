import numpy as np
import math
#distancia promedio, entre las distancias promedios de los centroides con sus datos asociados

def distCent(data, indexs, gruposCent, centroides, k):
    sumCentroide = lambda i : np.mean( np.linalg.norm( ( data[ indexs[ gruposCent == i], :] - centroides[i]), axis = 1, ord = 2)) 
    return np.mean([sumCentroide(indexCent) for indexCent in range(k)]) 




#data = datos
#index: vector numpy con la indeces mezclados
#k: cantidad de cluster
#tol: tolerancia de corte para el promedio de distancias entre los centroides y sus datos asociados

# el algoritmo corta por una tolerancia
# a partir de unos centroides elegidos aleatoriamente itera para armar los cluster(grupos)
# luego si finaliza porque los ya no hay cambio en los grupos, o el fin de iteraciones 
# calula la distancia promedio, para verificar si finalizar con esos grupos, y centroides o
# elegir nuevamente otro centroides aleatorios, y luego hacer la iteracion nuevamente

#output
# gruposCentroides: array numpy con indice donde indica a cual centroide pertenece el dato (ordenado con indexs)
# centroides : lista con numpy array de centroides

def k_medias_batch(data, indexs, k, centroides, iterMax = 200):

    
    H, W = data.shape
    
    gruposCentroides = np.zeros((H))
    flagCentVacio = False
    distancias = np.zeros((H,k)) 
        
    for n in range(iterMax): # buscar la convergencia con los centroides dados como parametros 
        for i, centroide in enumerate(centroides):
            distancias[:, i] = np.linalg.norm (data[indexs[:],:] - centroide, ord = 2, axis = 1)

        gruposCentroidesNew = np.argmin(distancias, axis = 1)
        
        # si hay algun centroide el cual no tiene asociado ningun dato, corta el algoritmo
        flagCentVacio = any(list([indexs[gruposCentroidesNew == i].shape[0] == 0 for i in range(k)]))
        if flagCentVacio:
            break
        
        if (gruposCentroidesNew == gruposCentroides).all(): #grupos iguales, se corta la iteracion 
            break

        gruposCentroides = gruposCentroidesNew
        centroides = [ np.mean( data[ indexs[ gruposCentroides == indexCent], :], axis = 0) for indexCent in range(k)]

    return  centroides, gruposCentroides, flagCentVacio

def k_medias_tol(data, indexs, k, tol = 0.1, iterMaxTol = 200, iterMaxConv = 200):
    
    H = data.shape[0]
    
    #funcion para elegir centroides aleatoriamente
    getCentroides = lambda : [ data[np.random.randint(0,H), :] for i in range(k)]
    
    centroidesMin = list() 
    gruposCentroidesMin = np.zeros((H))
    distanciaMin = math.inf
    
    for j in range(iterMaxTol):  # si no corta por la tolerancia, entoces lo hace por fin de iteracion
        
        centroides, gruposCentroides, flagCentVacio = k_medias_batch(data, indexs, k, getCentroides())
        
        if  flagCentVacio: # no dejar que exista ningun centroide vacio
            continue
            
        distancia = distCent(data, indexs, gruposCentroides, centroides, k)
        
        if distancia < distanciaMin: # guardar la mejor aproximacion de grupos posible
            distanciaMin = distancia
            gruposCentroidesMin = gruposCentroides
            centroidesMin = centroides
        
        if distancia < tol:
            break


    return  centroidesMin, distanciaMin, gruposCentroidesMin


def gauss_k_medias(data, centroides, k):
    
    H = data.shape[0]
    transfGauss = np.zeros((H,k))
    for i, centroide in enumerate(centroides):
        transfGauss[:, i] = np.exp(-0.5 * np.linalg.norm((data[:, :] - centroide), axis = 1) ** 2 )

    return transfGauss 




