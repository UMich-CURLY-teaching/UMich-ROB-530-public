#!usr/bin/env python

#
# MCMC (random-walk Metropolis-Hastings) example for sampling from a banana distribution
# http://www.stat.columbia.edu/~gelman/research/published/GelmanMeng1991
# Author: Fangtong Liu
# Date: 06/14/2020
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":

    # banana distribution, play with A, B, C1, and C2 to get different shapes
    A = 5
    B = 1
    C1 = 4
    C2 = 4


    def f(x1, x2):
        return np.exp(-0.5 * (A * np.power(x1, 2) * np.power(x2, 2) + np.power(x1, 2) + np.power(x2, 2)
                              - 2 * B * x1 * x2 - 2 * C1 * x1 - 2 * C2 * x2))


    X1, X2 = np.meshgrid(np.arange(-1, 6 + 0.01, 0.01), np.arange(-1, 6 + 0.01, 0.01))
    Z = f(X1, X2)


    # fig1 = plt.figure()
    # plt.pcolor(X1, X2, Z)
    # # plt.contour(X1, X2, Z)
    # plt.show()

    # MCMC
    rejected = 0
    accepted = 0
    Max_iter = 100000

    # proposal distribution
    r = np.zeros([2, 1])
    prop_cov = np.diag([1, 1])
    prop_L = np.linalg.cholesky(prop_cov)
    x = np.zeros([2, 1])
    sample = [x]

    for i in range(Max_iter):
        # sample from the proposal distribution to create a Markov chain
        x_new = x + np.dot(prop_L, np.random.randn(2, 1))

        # accept with probability A
        u = np.random.rand(1)
        A = min(1, f(x_new[0], x_new[1]) / f(x[0], x[1]))

        if A > u:  # accept
            x = np.copy(x_new)
            sample.append(x)
            accepted = accepted + 1
        else:
            rejected = rejected + 1
    sample = np.array(sample).reshape(-1, 2)

    # plot
    fig2 = plt.figure()
    # ax = fig2.gca()

    plt.hist2d(sample[:, 0], sample[:, 1], bins=100)
    # ax.set_xlim([-1, 6])
    # ax.set_ylim([-1, 6])
    plt.show()



