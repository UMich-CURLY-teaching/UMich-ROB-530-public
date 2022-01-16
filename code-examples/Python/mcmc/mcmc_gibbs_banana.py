#!usr/bin/env python

#
# MCMC (Gibbs sampling) example for sampling from a banana distribution
# http://www.stat.columbia.edu/~gelman/research/published/GelmanMeng1991
# Author: Fangtong Liu
# Date: 06/14/2020
#

import numpy as np
import matplotlib.pyplot as plt


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

    fig1 = plt.figure()
    plt.pcolor(X1, X2, Z)
    # plt.contour(X1, X2, Z)
    plt.show()

    # MCMC
    rejected = 0
    accepted = 0
    Max_iter = 100000

    # proposal distribution
    x2 = np.random.randn(1)
    sample = []

    for i in range(Max_iter):
        # sample from the proposal distribution to create a Markov chain
        # sample from p(x1)
        x1 = (B * x2 + C1) / (A * (x2 ** 2) + 1) + np.sqrt(1 / (A * (x2 ** 2) + 1)) * np.random.randn(1)
        # sample from p(x2 | x1)
        x2 = (B * x1 + C2) / (A * (x1 ** 2) + 1) + np.sqrt(1 / (A * (x1 ** 2) + 1)) * np.random.randn(1)

        sample.append(np.array([[x1], [x2]]))
    sample = np.array(sample).reshape(-1, 2)

    # plot
    fig2 = plt.figure()
    plt.hist2d(sample[:, 0], sample[:, 1], bins=100)
    plt.show()