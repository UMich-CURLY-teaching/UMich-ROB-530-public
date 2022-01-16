#!usr/bin/env python

#
# Robust Linear Regression via M-Estimation
# Author: Fangtong Liu
# Date: 06/14/2020
#

import numpy as np
import matplotlib.pyplot as plt


def basis(x1, x2):
    """compute basis (design) matrix"""
    s = 1.75  # bandwidth (length-scale)
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * s ** 2))


if __name__ == "__main__":

    # create a dataset
    z = np.arange(0, 8.1, 0.1)
    y = np.exp(-0.1 * z) * np.cos(z)  # true process

    # pick some training points randomly
    N = 15
    idx = np.unique(np.floor(np.random.rand(N, 1) * len(z)))
    N = len(idx)
    idx = idx.astype(int)
    x = z[idx]  # traiing input
    t = y[idx] + 0.05 * np.random.randn(N, 1).reshape(-1)  # noisy target

    # add a few outliers!
    outliers = np.unique(np.floor(np.random.rand(5, 1) * len(t)))
    outliers = outliers.astype(int)
    t[outliers] = t[outliers] + (2 * np.random.rand(len(outliers), 1) - 1).reshape(-1)

    # here we solve for w in linear model y = w^T * phi

    # compute basis (design) matrix
    Phi = np.zeros([len(x), len(x)+1])
    Phi[:, 0] = 1  # bias basis
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]-1):
            Phi[i, j+1] = basis(x[i], x[j])

    jitter = 1e-3 * np.eye(Phi.shape[1])  # to avoid singularity
    # solve for initial w using least squares with l_2 regularizer
    w_init = np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + jitter), Phi.T)
    w_init = np.dot(w_init, t)

    # now solve using IRLS
    max_iteration = 1000
    iter = 0
    B = np.eye(N)  # weights for || t - Phi * w || ^ 2
    r = np.zeros([N, 1])
    eps_termination = 1e-9  # termination threshold
    w = np.copy(w_init)
    w0 = np.copy(w)
    alpha = 0.1  # Cauchy loss parameter
    while iter < max_iteration:
        iter = iter + 1
        # compute weights
        for i in range(N):
            B[i, i] = 1 / (1 + ((t[i] - np.dot(Phi[i, :], w)) / alpha) ** 2)
        # normalize weights
        B = B / np.sum(np.sum(B))

        # solve normal equations
        w = np.dot(np.dot(Phi.T, B), Phi) + jitter
        w = np.dot(np.linalg.inv(w), np.dot(np.dot(Phi.T, B), t))

        if not np.mod(iter, 10):
            print('iteration: ', iter)
            print('|| w ||: ', np.linalg.norm(w))

        # check if converged
        if np.linalg.norm(w - w0) < eps_termination:
            print('Converged')
            break
        else:
            w0 = np.copy(w)

    # predict at arbitrary inputs
    Phi_test = np.zeros([len(z), len(w)])
    Phi_test[:, 0] = 1  # bias basis
    for i in range(Phi_test.shape[0]):
        for j in range(Phi_test.shape[1]-1):
            Phi_test[i, j+1] = basis(z[i], x[j])

    y_test = np.dot(Phi_test, w)  # predict all at once

    ##################### plotting ##################
    # plot the true process and training points
    fig1 = plt.figure()
    plt.plot(z, y, linewidth=3, label='true process')
    plt.plot(x, t, '.', markersize=10, label='training points')
    plt.plot(z, y_test, '--', color='k', linewidth=3, label='prediction')
    plt.legend(loc='best')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y(x)$')
    plt.xlim([0, 8])
    plt.ylim([-1.25, 1.25])
    # fig1.savefig('lin_reg_m_estimation.png')
    plt.show()

    fig2 = plt.figure()
    plt.stem(w, use_line_collection=True)
    plt.xlabel('basis')
    plt.ylabel('w')
    # fig2.savefig('lin_reg_w_m_estimation.png')
    plt.show()

