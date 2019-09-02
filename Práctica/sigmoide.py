# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:44:54 2019

@author: Messi
"""
from math import e
import numpy as np

def sigmoide(vLineal, b):
    y = 2/(((1 + e**(-(b*vLineal)))) + 1)
    return y