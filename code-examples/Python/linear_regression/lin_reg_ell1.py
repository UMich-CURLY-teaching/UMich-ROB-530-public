#!usr/bin/env python

#
# Robust Linear Regression with l1-regularizer
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
    N = 15  # number of training points
    idx = np.unique(np.floor(np.random.rand(N, 1) * len(z)))
    N = len(idx)
    idx = idx.astype(int)
    x = z[idx]  # training input
    t = y[idx] + 0.05 * np.random.randn(N, 1).reshape(-1)  # noisy target

    # add a few outliers
    outliers = np.unique(np.floor(np.random.rand(5, 1) * len(t)))  # random indicies
    outliers = outliers.astype(int)
    t[outliers] = t[outliers] + (2 * np.random.rand(len(outliers), 1) - 1).reshape(-1)

    # here solve for w in a linear model y = w^T * phi
    # compute vasis (design) matrix
    Phi = np.zeros([len(x), len(x)+1])
    Phi[:, 0] = 1  # bias basis
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]-1):
            Phi[i, j+1] = basis(x[i], x[j])

    # solve for initial w using least squares with l_2 regularizer
    w_init = np.dot(Phi.T, Phi) + 0.1 * np.eye(Phi.shape[1])
    w_init = np.dot(np.linalg.inv(w_init), np.dot(Phi.T, t))

    # now solve the ell_1 problem using IRLS
    max_iteration = 1000
    iter = 0
    B = np.eye(N)  # weights for ||t - Phi * w||^2
    G = np.eye(len(w_init))  # weights for ||w||^2
    r = np.zeros([N, 1])
    delta = 1e-6  # to avoid division by zero
    eps_termination = 1e-9  # termination threshold
    w = np.copy(w_init)
    w0 = np.copy(w)

    while iter < max_iteration:
        iter = iter + 1
        # compute weights
        G[0, 0] = 1 / max(delta, np.abs(w[0]))
        for i in range(N):
            B[i, i] = 1 / max(delta, np.abs(t[i] - np.dot(Phi[i, :], w)))
            G[i+1, i+1] = 1 / max(delta, np.abs(w[i+1]))
        # normalize weights
        B = B / np.sum(np.sum(B))
        G = G / np.sum(np.sum(G))

        # solve normal equations
        w = np.dot(np.dot(Phi.T, B), Phi) + 0.1 * G
        w = np.dot(np.linalg.inv(w), np.dot(np.dot(Phi.T, B), t))

        if not np.mod(iter, 10):
            print('iteration: ', iter)
            print('||w||: ', np.linalg.norm(w, 0))

        # check if converged
        if np.linalg.norm(w-w0, 0) < eps_termination:
            print('Converged')
            break
        else:
            w0 = np.copy(w)

    # find survived basis for plotting, ignoring bias
    sb = np.abs(w[1:]) > delta

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
    plt.plot(x[sb], t[sb], 'o', fillstyle='none', color='k', markersize=15, linewidth=3, label='survived basis')
    plt.legend(loc='best')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y(x)$')
    plt.xlim([0, 8])
    plt.ylim([-1.25, 1.25])
    # fig1.savefig('lin_reg_ell1.png')
    plt.show()

    fig2 = plt.figure()
    plt.stem(w, use_line_collection=True)
    plt.xlabel('basis')
    plt.ylabel('w')
    # fig2.savefig('lin_reg_w_ell1.png')
    plt.show()



