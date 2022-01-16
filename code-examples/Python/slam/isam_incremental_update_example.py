#!usr/bin/env python

import numpy as np
from QR import QR_function

if __name__ == "__main__":
    print('Random Measurement Matrix:')
    A = np.random.randn(10, 4)
    print(A)

    print('QR Factorization Using Givens Rotations:')
    Q, R = QR_function(A)
    print('Q\n', Q, '\nR\n', R)

    print('Note that R is indeed equal to the information matrix square root (up to sign flip of rows)')
    R_chol = np.linalg.cholesky(np.dot(A.T, A)).T
    print(R_chol)

    print('Here is the transformed measurement matrix:')
    At = np.dot(Q.T, A)
    At[abs(At) < abs(2*np.finfo(float).eps)] = 0
    print(At)

    print('Suppose now that we ass a new measurement row:')
    At_aug = np.vstack((At, np.array([0, 0, 0, 1])))
    print(At_aug)

    print('We can incrementally obtain the new factorization using one Givens rotation:')
    alpha = At_aug[3, 3]
    beta = At_aug[10, 3]
    if beta == 0:
        print('beta = 0')
        c = 1
        s = 0
    elif abs(beta) > abs(alpha):
        print('|beta| > |alpha|')
        c = -alpha / beta / np.sqrt(1 + (alpha / beta)**2)
        s = 1 / np.sqrt(1 + (alpha / beta)**2)
    else:
        print('Otherwise')
        c = 1 / np.sqrt(1 + (alpha / beta) ** 2)
        s = -beta / alpha / np.sqrt(1 + (alpha / beta) ** 2)
    G = np.eye(11)
    G[3, 3] = c
    G[3, 10] = s
    G[10, 3] = -s
    G[10, 10] = c
    print(G)

    print('Incrementally applying the Givens rotation to our augmented system:')
    gAt_aug = np.dot(G.T, At_aug)
    print(gAt_aug)

    print('Compare this with the batch QR factorization of At_aug:')
    Qt_aug, Rt_aug = np.linalg.qr(At_aug)
    print('Qt_aug\n', Qt_aug, '\nRt_aug\n', Rt_aug)

    print('And also compare with the Cholesky factorization of the augmented system:')
    R_aug_chol = np.linalg.cholesky(np.dot(At_aug.T, At_aug)).T
    print(R_aug_chol)

