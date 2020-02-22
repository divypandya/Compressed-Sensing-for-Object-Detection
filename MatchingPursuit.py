# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:46:54 2019

@author: Divy Pandya
"""

import numpy as np

def MP(A, b, k):
    r = b
    x = np.zeros(shape = (A.shape[1], 1))
    
    for s in range(k):
        i = np.argmax(np.abs(A.T @ r))
        x[i] = x[i] + A[:,i].T @ r
        r = b - A @ x
    
    return x, r