# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:55:08 2019

@author: Divy Pandya
"""
import numpy as np
def omp(A, b, k, tol = 0):
    # OMP Solve the P0 problem via OMP
    #
    # Solves the following problem:
    #   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
    #
    # The solution is returned in the vector x
        
    # Initialize the vector x
    r = b
    x = np.zeros(shape = (A.shape[1], 1))
    support = []
    s = 0
    while np.linalg.norm(r, 2) > tol and s < k:
        i = np.argmax(np.abs(A.T @ r))
        support.append(i)
        As = A[:, support]
        x[support] = np.linalg.lstsq(As, b, rcond = None)[0]
        r = b - A@x;
        s+=1
    return x

