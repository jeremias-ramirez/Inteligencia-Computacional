import numpy as np

def membresia(x, vConj, tipo):
    u = 0.0
    if tipo == 1:
        if x < vConj[0]  or vConj[3] < x:
            u = 0
        elif vConj[0] <= x and x < vConj[1] :
            u = (x - vConj[0]) / (vConj[1] - vConj[0])
        elif vConj[1] <= x and x <= vConj[2] :
            u = 1
        elif vConj[2] < x and x <= vConj[3]:
            u = 1 - (x - vConj[2]) / (vConj[3] - vConj[2])

    elif tipo == 2:
        u = np.exp(-0.5 * ((x - vConj[0]) / vConj[1]) ** 2) 
    return u

def matrizMembresia(xV, M, tipo):
    y = np.zeros((M.shape[0], xV.shape[0])) 
    y = np.array( list( map( lambda me: list(map( lambda xe: membresia(xe, me, tipo), xV)), M)) )
    return y

def matrizMembresia1E(x, M, tipo):
    return matrizMembresia(np.array([x]), M, tipo).reshape((M.shape[0]))


def areaCentr_Trapezoide(h, vConj):
    if h == 0:
       return (0,0) 
    area = np.zeros((3,1))
    centr = np.zeros((3,1))
    if vConj[0] != vConj[1]:
        area[0] = abs(vConj[1] - vConj[0]) * h / 2
        centr[0] = vConj[0] + 2/3 * abs(vConj[1] - vConj[0])
    if vConj[1] != vConj[2]:
        area[1] = abs(vConj[2] - vConj[1]) * h 
        centr[1] = vConj[1] + 1/2 * abs(vConj[2] - vConj[1])
    if vConj[2] != vConj[3]:
        area[2]  = (vConj[3] - vConj[2]) * h / 2
        centr[2] = vConj[2] + 1/3 * abs(vConj[3] - vConj[2])
    
    return h * area.sum(), ((centr.T @ area)[0][0] / h * area.sum())

def areaCentr_Gauss(h, vConj):
    if h == 0.0:
       return (0,0)
    area = h * vConj[1] * np.sqrt(2 * np.pi)    
    return area, vConj[0] 
 
def defuzzificacion(gA, M, tipo):
    areaCentr = list() 
    if tipo == 1:
        areaCentr = [areaCentr_Trapezoide(gA[i], conj) for i, conj in enumerate(M)]
    else:
        areaCentr = np.array([areaCentr_Gauss(gA[i], conj) for i, conj in enumerate(M)])
    areaCentr = np.array(areaCentr)

    return (areaCentr[:,1].T @ areaCentr[:,0]) / areaCentr[:,0].sum()


def sistemaBorroso(x, r, M, S, tipo):
    gA = matrizMembresia1E(x, M, tipo)
    gAN = np.zeros((r.shape[0]))
    gAN[r-1] = gA
    return defuzzificacion(gAN, S, tipo)

def sistemaBorrosoMap(x, r, M, S, tipo):
    return np.array( list( map( lambda xe: sistemaBorroso(xe, r, M, S, tipo), x)))
