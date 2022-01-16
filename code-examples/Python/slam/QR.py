#!usr/bin/env python

#
# QR Factorization
# Author: Fangtong Liu
# Date: 04/27/2020
#

import numpy as np
from scipy import sparse


def QR_function(A):
    # Input:
    #   A:  A is a M*N matrix
    # Output:
    #   Q:  Q is a M*M orthogonal matrix
    #   R:  R is a M*N upper triangular matrix above the diagonal

    m, n = np.shape(A)
    QT = np.eye(m)
    for k in range(n):
        for i in range(k, m):
            if A[i, k] == 0:
                continue

            # Givens rotation matrix
            alpha = A[k, k]
            beta = A[i, k]
            if beta == 0:
                c = 1
                s = 0
            elif np.abs(beta) > np.abs(alpha):
                c = -alpha / beta / np.sqrt(1 + (alpha / beta)**2)
                s = 1 / np.sqrt(1 + (alpha / beta)**2)
            else:
                c = 1 / np.sqrt(1 + (alpha / beta) ** 2)
                s = -beta / alpha / np.sqrt(1 + (alpha / beta) ** 2)

            G = np.eye(m)
            G = sparse.lil_matrix(G)
            G[k, k] = c
            G[k, i] = s
            G[i, k] = -s
            G[i, i] = c

            QT = G.T.dot(QT)
            A = G.T.dot(A)

            if abs(A[i, k]) < abs(np.finfo(float).eps):
                A[i, k] = 0

    Q = QT.T
    R = A

    return Q, R


