#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/31/2020
#

import numpy as np
from unscented_transform import unscented_transform


# unscented_transform construct at instance of this class
if __name__ == "__main__":

    # create a random mean and covariance
    n = 3
    x = np.random.randn(3, 1)
    L = np.random.randn(3, 3)
    P = np.dot(L, L.T)

    A = np.random.randn(2, 3)

    def f(z):
        return np.dot(A, z)

    # propagate the uncertainty using UT and affine model to compare
    kappa = 2
    ut = unscented_transform(x, P, f, kappa)
    ut.propagate()

    print('kappa = ', kappa)
    print('norm(ut.mean - A*x) = ', np.linalg.norm(ut.mean - np.dot(A, x).reshape(-1)))
    temp = ut.Cov - np.dot((np.dot(A, P)), A.T)
    print('norm(ut.Cov - A*P*A.T) = ', np.linalg.norm(temp))
    
    kappa = 1
    ut.set(x, P, f, kappa)
    ut.propagate()

    print('--------------------------------------------------------------')
    print('kappa = ', kappa)
    print('norm(ut.mean - A*x) = ', np.linalg.norm(ut.mean - np.dot(A, x).reshape(-1)))
    temp = ut.Cov - np.dot((np.dot(A, P)), A.T)
    print('norm(ut.Cov - A*P*A.T) = ', np.linalg.norm(temp, 'fro'))

    kappa = 0
    ut.set(x, P, f, kappa)
    ut.propagate()

    print('--------------------------------------------------------------')
    print('kappa = ', kappa)
    print('norm(ut.mean - A*x) = ', np.linalg.norm(ut.mean - np.dot(A, x).reshape(-1)))
    temp = ut.Cov - np.dot((np.dot(A, P)), A.T)
    print('norm(ut.Cov - A*P*A.T) = ', np.linalg.norm(temp, 'fro'))




