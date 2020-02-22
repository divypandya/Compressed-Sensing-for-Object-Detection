# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:21:14 2019

@author: Divy Pandya
"""
import time
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.linear_model import orthogonal_mp
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
class MOD(object):
    def __init__(self, n_components, max_iter = 30, tol = 1e-4, n_nonzero = None):
       self.dictionary = None
       self.n_components = n_components
       self.max_iter = max_iter
       self.tol  = tol
       self.n_nonzero = n_nonzero
         
    def _intialize(self, y):
        u, s, v = linalg.svd(y)
        rows, cols = u.shape
        if self.n_components <= cols:
            self.dictionary = u[:, :self.n_components]
        else:
            self.dictionary = np.c_[u, np.zeros((rows, self.n_components - cols))]
    
    def _update_dict(self, y, d):
        t0 = time.time()
        
        for i in range(self.max_iter):
            a = orthogonal_mp(d, y, n_nonzero_coefs = self.n_nonzero)
            e = linalg.norm(y - np.dot(d, a))
            
            dt = (time.time() - t0)
            print('Iteration %3d error: %.4f (elapsed time: %ds)' %(i+1, e, dt))
            
            if e < self.tol:
                break
            
            d = y.dot(a.T).dot(linalg.pinv(a.dot(a.T)))
            d = normalize(d, axis = 0)
            
        return d
    
    def fit(self, sample):
        check_array(sample)
        self._intialize(sample)
        self.dictionary = self._update_dict(sample, self.dictionary)
        return self
    
    def tranform(self, sample):
        return orthogonal_mp(self.dictionary, sample, n_nonzero_coefs = self.n_nonzero)
    

if __name__ == '__main__':
    im_ascent = scipy.misc.ascent().astype(np.float)
    model = MOD(600)
    model.fit(im_ascent)
    dictionary = model.dictionary
    A = model.tranform(im_ascent)
    im_reconstruct = dictionary.dot(A)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_ascent)
    plt.subplot(1, 2, 2)
    plt.imshow(im_reconstruct)
    plt.show()
    
    