import numpy as np

def getCentroidesGrupos(data, grupos):
    #print(data[grupos[0],:])
    _, W = data.shape
    getMean = lambda grupo : np.mean( data[grupo , : ], axis = 0)
    centroides = list( map( getMean, grupos ) )
    #print(centroides) 
    #input()
    return centroides

def reasigacionGrupos(data, centroides, k):
    grupos = [list() for i in range(k)]

    for i in range( len(data[:,0]) ):
        #print(i) 
        #print(data[i, :].shape)
        #print( "cnea")
        #print(list(map(lambda cn: cn.shape, centroides)))
        #input()
        distancias = np.linalg.norm((data[i, :] - centroides), axis = 1)**2
        #print(distancias) 
        index = np.argmin(distancias, axis = 0)
        #print(index)

        #input()
        grupos[index].append(i)
    
    return grupos 

def indexArgMin(xl, centroides):
    distancias = np.linalg.norm((xl - centroides), axis = 1)**2
    return np.argmin(distancias, axis = 0)

def cmpGrupos(grupos1, grupos2):
    flag = True
    for i, grupo1 in enumerate(grupos1):
        if grupo1 != grupos2[i]:
            flag = False
            break
    return flag

def updateGroups(data, grupos, k):
    flag = True 
    iterMax = 1000
    for i in range(iterMax):
        centroides = getCentroidesGrupos(data, grupos) 
        #print("gropo ol")
        #print(grupos)
        gruposN = reasigacionGrupos(data, centroides, k)
        #print("grupo n")
        #print(gruposN)
        #input()
        cmpGroups = [ grupos[i] != gruposN[i] for i in range(k)]
        flag = any(cmpGroups);
        if not flag:
            break
        grupos.clear()
        grupos = gruposN.copy()
        gruposN.clear()
    return grupos 
    

def k_media_batch(data, k):
    
    H, _ = data.shape
    tamG = int(H/k) # tamaño del grup
    indexs = np.arange(0, H, 1)
    np.random.shuffle(indexs)
    
    centroides = [ data[indexs[i]] for i in range(k) ]

    gruposI = reasigacionGrupos(data, centroides, k)

    return updateGroups(data, gruposI, k) 


#def gradienteJ(data, centroides)
#
#def k_medias_online(data, k):
#    H, _ = data.shape
#    tamG = int(H/k) # tamaño del grup
#    indexs = np.arange(0, H, 1)
#    np.random.shuffle(indexs)
#    centroides = [ data[indexs[i]] for i in range(k) ]
#    vel = 0.1
#    distMin = 0.001
#    dist = 1.0
#    while distMin < dist :
#        for i in range(len (H) ):
#            index = indexArgMin(data[i, :], centroides)
#            centroides[index] = centroides[index] + vel * (data[i, :] - centroides[index])
#            dist = gradienteJ(data, centroides)
#            
#            if distMin < distMin:
#                break
#
