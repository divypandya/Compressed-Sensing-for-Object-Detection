# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:51:02 2019

@author: Divy Pandya
"""

import time
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.linear_model import orthogonal_mp

class KSVD(object):
    def __init__(self, n_components, max_iter = 30, tol = 1e-4, n_nonzero = None):
        self.dictionary = None
        self.code = None
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_nonzero = n_nonzero
        
    def _initialize(self, y):
        u, s, v = linalg.svd(y)
        rows, cols = u.shape
        if self.n_components <= cols:
            self.dictionary = u[:, :self.n_components]
        
        else:
            self.dictionary = np.c_[u, np.zeros((rows, self.n_components - cols))]
            
    def _update_dict(self, y, d, x):
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            
            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices = False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0]*v[0, :]
            
        return d, x
    
    def fit(self, sample):
        self._initialize(sample)
        t0 = time.time()
        for i in range(self.max_iter):
            x = orthogonal_mp(self.dictionary, sample,
                              n_nonzero_coefs = self.n_nonzero)
            #e = np.linalg.norm(sample - np.dot(self.dictionary, x))
            e = np.sqrt(np.square(sample - np.dot(self.dictionary, x)).mean())
            
            dt = (time.time() - t0)
            print("Iteration %3d error: %.4f (elapsed time: %ds)" %(i+1, e, dt))
            
            if e < self.tol:
                break
            
            self.dictionary, x =  self._update_dict(sample, self.dictionary, x)
            
        self.code = orthogonal_mp(self.dictionary, sample,
                                  n_nonzero_coefs = self.n_nonzero)
        
        return self.dictionary, self.code
    
    
if __name__ == '__main__':
    im_ascent  = scipy.misc.ascent().astype(np.float)
    model = KSVD(600)
    dictionary, code = model.fit(im_ascent)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_ascent)
    im_recon = dictionary.dot(code)
    plt.subplot(1, 2, 2)
    plt.imshow(im_recon)
    plt.show()