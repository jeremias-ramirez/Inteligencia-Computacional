# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:58:39 2019

@author: Messi
"""
import numpy as np

def som():
    datos = np.genfromtxt("files/circulo.csv")
    cantDatos,cantPesos = np.shape(datos)
    filasMapa = 2
    columnasMapa = 2
    mapa = ceil(filasMapa,columnasMapa)
    distancias = np.zeros(filasMapa,columnasMapa)
    t_ap = 0.1 #inicio    

    #modificar para cambiar las epocas e intervalos con los que se modifican R y t_ap
    R = 1
    cant_epocas = 1000
    modificador_epocas_min = 1
    modificador_epocas_max = 400 
#   Pela
#   Para R
    modificador_epocas_aux = 0
    modificador_epocas = 4
    
    #Para t_ap
    t_apMax = 0.1
    t_apMin = 0.01
    epoc_min = 300
    epoc_max = 700
    
    #Para Graficar con el mesh
    grillaX = np.zeros((1,filasMapa*columnasMapa))
    grillaY = np.zeros((1,filasMapa*columnasMapa))
    grillaZ = np.zeros((filasMapa*columnasMapa,filasMapa*columnasMapa))
    
    #creo el mapa con los pesos al azar
    for i_filas in range(filasMapa):
        for i_columnas in range(columnasMapa):
            mapa(i_filas,i_columnas) = np.random(1,cantPesos)-0.5;  #creo el mapa con los pesos

#    fig = figure(1)
#    %plotSOM(mapa, cantPesos);
#    dibujarCirculo_T(1);
#    %graficarSOM(mapa,filasMapa,columnasMapa,grillaX,grillaY,grillaZ);
#    pause #epero que el usuario de enter para empezar a entrenar el mapa
    
    it = 1
    for epoc in range(cant_epocas):
               
#        if(epoc==it):
#            clf(fig)
#            clf(fig,reset)
#            clf reset
#            plotSOM(mapa, cantPesos); 
#            dibujarCirculo_T(1);
#            graficarSOM(mapa,filasMapa,columnasMapa,grillaX,grillaY,grillaZ);
#            sllep(0.2) #espero pa siguiente epoca
#            it = it + 1

            #-------------------modificacion t_ap y R----------------

        modificador_epocas_aux = fix(epoc/100) #Saco el entero de la division
        if(modificador_epocas_aux>modificador_epocas && epoc<=modificador_epocas_max && R!=0):
            R = R-1
            modificador_epocas = modificador_epocas_aux
        else if (epoc>modificador_epocas_max):
                R = 1 #Despues de la 400, si es que la maxima, que solo queda la ganadora

        if(epoc>=epoc_min && epoc<=epoc_max):
#            Ecuacion de la recta lineal donde y es la taza de #aprendizaje, x lasepocas
            t_ap = ((epoc*(t_apMin-t_apMax))/(epoc_max-epoc_min))-(epoc_min*(t_apMin-t_apMax))/(epoc_max-epoc_min)+t_apMax;            
        else if(epoc>epoc_max)
            t_ap = t_apMin

            #-------------------para cada patron----------------

        for i in range(cantDatos):
            patron = datos(i,:)
            #calculo las distancias del patron i a todo el mapa.
            for ind_filasMapa in range(filasMapa):
                for ind_columnasMapa in range(columnasMapa):
                    aux = mapa{ind_filasMapa,ind_columnasMapa}
                    distancias(ind_filasMapa,ind_columnasMapa) = np.linalg.norm(patron-aux)
            
            #busco el mÃ­nimo de todo el mapa: neurona ganadora
            m1,v_filas = min(distancias)
            aux,col = min(m1) #col tiene el indice de la columna de valor minimo
            fil = v_filas(col) #fil tiene el indice de la fila de valor minimo        
            

            # Actualizacion del mapa
            for i_a in range(filasMapa): #i_a = i de actualizacion
                alfa = abs(i_a-fil)
                if(alfa<=R):
                    
                    gama=R-alfa;
                    if(col-gama<=0 && col+gama<=columnasMapa):
                        for i_c in range(col+gama):
                            aux=mapa{i_a,i_c}
                            mapa(i_a,i_c)=aux+t_ap*(patron-aux)
                    else if(col+gama>=columnasMapa && col-gama>0):
                        for i_c=(col-gama) in range(columnasMapa):
                            aux = mapa{i_a,i_c};
                            mapa(i_a,i_c) = aux+t_ap*(patron-aux);
                    
                    else if (col+gama>=columnasMapa && col-gama<=0):
                        for i_c in range (columnasMapa):
                            aux = mapa{i_a,i_c};
                            mapa(i_a,i_c)=aux+t_ap*(patron-aux);
                        
                    else
                        for i_c=col-gama in range(col+gama):
                            aux=mapa{i_a,i_c};
                            mapa(i_a,col-gama:i_c)=aux+t_ap*(patron-aux);
            
#            
#            %xAct=fil-R:fil+R;
#            %vec_indActFil=xAct(xAct>=1 & xAct<=filasMapa);
#            %i_c=0;
#            %for indAct=1:size(vec_indActFil,2);
#            %    yAct=col-i_c:col+i_c;
#            %    vec_indActCol=yAct(yAct>=1 & yAct<=columnasMapa);
#            %    for indActCol=1:size(vec_indActCol,2);
#            %        aux=mapa{indAct,vec_indActCol(1,indActCol)};
#            %        mapa(indAct,vec_indActCol(1,indActCol))=aux+t_ap*(patron-aux);
#            %    endfor
#            %    if (i_c==R)
#            %        i_c=i_c+1;
#            %    else
#            %        i_c=i_c-1;
#            %    endif
#            %endfor

        epoc
