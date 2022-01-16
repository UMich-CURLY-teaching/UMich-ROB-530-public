#!/usr/bin/env python
#
# This function is intended to be used as 'cholupdate' in MATLAB
# Author: Fangtong Liu
# Date: 05/10/2020
#

import numpy as np


def givens(a, b):
    # find a Givens rotation
    if b == 0:
        c = 1
        s = 0
    elif abs(b) > abs(a):
        tau = - a / b
        s = 1 / np.sqrt(1 + tau ** 2)
        c = s * tau
    else:
        tau = - b / a
        c = 1 / np.sqrt(1 + tau ** 2)
        s = c * tau
    return np.array([[c, -s], [s, c]])


def cholupdate(Rold, x):
    # Cholesky update, uding GIvens rotations
    Rold = Rold.T
    n = Rold.shape[0]
    R = np.hstack((Rold, x.reshape(n, 1)))

    for k in range(n):
        g = givens(R[k, k], R[k, n])
        R[:, [k, n]] = np.dot(R[:, [k, n]], g.T)
        for i in range(n):
            if R[i, k] > 0:
                R[i, k] = -R[i, k]
        R[k, k] = abs(R[k, k])
    return R[:, 0:n].T
