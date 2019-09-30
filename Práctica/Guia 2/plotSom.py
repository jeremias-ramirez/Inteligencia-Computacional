# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:41:46 2019

@author: Messi
"""
import numpy as np

def plotSOM(m_pesos,dim_pesos):
    
    dim_map = size(m_pesos)
    x = zeros(dim_pesos,1)
    y = zeros(dim_pesos,1)
    
    for i_filas in range(dim_map(1)):
        for i_columnas in range(dim_map(2)-1):
            x(1)=m_pesos{i_filas,i_columnas}(1,1)
            x(2)=m_pesos{i_filas,i_columnas+1}(1,1)
            y(1)=m_pesos{i_filas,i_columnas}(1,2)
            y(2)=m_pesos{i_filas,i_columnas+1}(1,2)

            plot(x,y)

    
    for i_columnas in range(dim_map(2)):
        for i_filas in range(dim_map(1)-1):
            x(1)=m_pesos{i_filas,i_columnas}(1,1)
            x(2)=m_pesos{i_filas+1,i_columnas}(1,1)
            y(1)=m_pesos{i_filas,i_columnas}(1,2)
            y(2)=m_pesos{i_filas+1,i_columnas}(1,2)

            plot(x,y)
