# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:50:58 2019

@author: Divy Pandya
"""

import numpy as np

# Moore-Penrose Pseudoinverse
def MPP(A):
    u, s, vh = np.linalg.svd(A)
    sinv = 1./s
    Dinv = np.diag(sinv)
    Dinv = np.concatenate((Dinv, np.zeros((A.shape[1], A.shape[0] - s.shape[0]))), axis = 1)
    U_T, V = u.T, vh.T
    return V@Dinv@U_T

