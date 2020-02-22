# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:35:03 2019

@author: Divy Pandya
"""
import numpy as np
import OMP
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

n = 50
m = 100
s_max = 15
min_coeff_val = 1
max_coeff_val = 3
num_realizations = 200
base_seed = 4321
A = np.random.randn(n, m)
A_normalized = normalize(A, axis = 0)
eps_coeff = 1e-4
L2_error = np.zeros((s_max, num_realizations))

for s in range(1, s_max+1):
    
    for exp in range(1, num_realizations+1):
        x = np.zeros((m, 1))
        true_supp = np.random.permutation(m)[:s]
        x[true_supp, 0] = np.random.uniform(min_coeff_val, max_coeff_val, s)
        b = A_normalized.dot(x)
        
        x_omp = OMP.omp(A_normalized, b, s)
        x_omp[np.abs(x_omp) < eps_coeff] = 0
        
        L2_error[s-1, exp-1] = np.sum(np.square(x_omp - x))/np.sum(np.square(x))
        
        estimated_supp = np.sum(np.abs(x_omp) > eps_coeff)
        

plt.figure()
plt.plot(range(1, s_max+1), np.mean(L2_error, 1))
plt.xlabel("cardinality of the solution")
plt.ylabel("L2_error")
plt.show()