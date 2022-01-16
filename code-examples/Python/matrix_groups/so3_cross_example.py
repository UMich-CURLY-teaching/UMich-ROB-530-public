#!usr/bin/env python

#
# An example for showing the relation between SO(3) Lie algebra and its
# generators with the usual cross product on R^3
#
# Author: Fangtong Liu
# Date: 05/17/2020
#

import numpy as np


def skew(x):
    # vector to skew R^3 -> so(3)
    matrix = np.array([[0, -x[2], x[1]],
                       [x[2], 0, -x[0]],
                       [-x[1], x[0], 0]])
    return matrix


def unskew(A):
    # so(3) -> R^3
    return np.array([[A[2, 1]], [A[0, 2]], [A[1, 0]]])


def bracket(A, B):
    return np.dot(A, B) - np.dot(B, A)


if __name__ == "__main__":

    # create random vectors and their cross product
    a = np.floor(4 * np.random.randn(3, 1) - 2)
    b = np.floor(4 * np.random.randn(3, 1) - 2)
    c1 = np.cross(a.T, b.T)

    # R^3 standard basis
    e1 = np.array([[1], [0], [0]])
    e2 = np.array([[0], [1], [0]])
    e3 = np.array([[0], [0], [1]])

    # so(3) (SO(3) Lie algebra) infinitesimal generators
    G1 = skew(e1)
    G2 = skew(e2)
    G3 = skew(e3)

    # reproduce the same cross product using skew(a) * b = a x b
    c2 = np.dot(skew(a), b)

    # now try the application of each generator independently and sum them
    c3 = a[0] * np.dot(G1, b) + a[1] * np.dot(G2, b) + a[2] * np.dot(G3, b)

    print('\na x b\t\tskew(a) * b\t\tsum(a(i) * G_i * b)')
    for i in range(3):
        # to adjust the spacing for negative sign
        if c1[0, i] < 0:
            print(c1[0, i], '\t\t', c2[i][0][0], '\t\t', c3[i][0][0])
        else:
            print('', c1[0, i], '\t\t ', c2[i][0][0], '\t\t ', c3[i][0][0])

    # now let's try playing with the Lie bracket! We take the Lie bracket of
    # any two generators and get the third generator back. This is similar
    # cross product of any two R^3 standard basis
    print('\n---------------------------------------------\n')
    print('unskew(bracket(G1, G2)) - e1 x e2 = ')
    print(unskew(bracket(G1, G2)).reshape(-1, 1) - np.cross(e1.T, e2.T).reshape(-1, 1))

    print('unskew(bracket(G2, G3)) - e2 x e3 = ')
    print(unskew(bracket(G2, G3)).reshape(-1, 1) - np.cross(e2.T, e3.T).reshape(-1, 1))

    print('unskew(bracket(G3, G1)) - e3 x e1 = ')
    print(unskew(bracket(G3, G1)).reshape(-1, 1) - np.cross(e3.T, e1.T).reshape(-1, 1))
