import numpy as np
from functools import reduce
#retorna el una lista de los porcentajes correspondiente a cada elemento
#el primer elemento va desde 0 hasta su porcentaje correspondiente, el segundo
#desde el porcentaje del primero mas su porcentaje correspondiente, ...
#el ultimo desde el porcentaje anterior acumulado hasta 1.0
def getRuleta(fitnessV):
    total = np.sum(fitnessV)

    def getRuletaAux(accum, elem):
        if not accum :
            accum.append(elem/total)
        else :
            accum.append(accum[-1] + elem/total)
        return accum

    return np.array(reduce(getRuletaAux, fitnessV, []))


#padres: matriz de dos filas (padres) y columna de longitud del cromosoma
# uC: umbral de la probalbilida de cruza[0,1]
def cruza(padres, uC):

    hijos = np.copy(padres)
    p = np.random.random(0,1)
    if p < uC : 
        rangeI = 0
        rangeF = padres.shape[1]
        ptoCruza = np.random.randint(rangeI, rangeF)
        hijos[:, ptoCruza:rangeF] = padres[::-1, ptoCruza:rangeF] #::-1 hace que se recorra al reves

    return hijos

#cromosoma : vector de filas (tam, )
#uM : umbral de la probabilidad de mutacion [0,1]
def mutacion(cromosoma, uM):
    p = np.random.random(0,1)
    mutado = np.copy(cromosoma)
    if p < uM:
        rangeI = 0
        rangeF = cromosoma.shape[0]

        ptoMutacion = np.random.randint(rangeI, rangeF)
        mutado[ptoMutacion] = int( not( int( cromosoma[ptoMutacion] )))
    return mutado

def getIndexRuleta(ruleta, pGiro, index):
    if ruleta[index] > pGiro:
        return index
    else:
        return getIndexRuleta(ruleta, pGiro, index + 1)

    
def seleccion(cntPadres, poblacion, ffitness):

    fitnessV = np.array( list( map( ffitness, poblacion )))
    elite = poblacion[np.argmax(fitnessV)]
    ruleta = getRuleta(fitnessV)

    padres = [poblacion[getIndexRuleta(ruleta, np.random.random(), 0)] for i in range(cntPadres - 1)]
    padres.append(elite)

    return np.array(padres)

def algoritmoGenetico(poblacion, ffitness, pC = 0.01, pM = 0.001, pBG = 0.1):
    return  

    
