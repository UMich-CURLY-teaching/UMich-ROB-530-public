#!usr/bin/env python

# Linear regression with l2-regularizer
#
# Author: Fangtong Liu
# Date: 06/13/2020
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
    y = np.exp(-0.1 * z) * np.cos(z)

    # pick some training points rasndomly
    N = 15  # number of training points
    idx = np.unique(np.floor(np.random.rand(N, 1) * len(z)))
    # temp = np.array([0.1, 0.5, 0.3, 0.1, 0.8])
    # u, idx = np.unique(1 + np.floor(temp * len(z)), return_index=True)
    N = len(idx)
    idx = idx.astype(int)
    x = z[idx]  # training input
    t = y[idx] + 0.05 * np.random.randn(N, 1).reshape(-1)  # noisy target

    # here we solve for w in a linear model y = w^T * phi
    # compute basis (design) matrix
    Phi = np.zeros([len(x), len(x) + 1])
    Phi[:, 0] = 1  # bias basis
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]-1):
            Phi[i, j+1] = basis(x[i], x[j])

    # solve for w using least squares with l_2 regularizer
    w = np.dot(Phi.T, Phi) + 0.1 * np.eye(Phi.shape[1])
    w = np.dot(np.linalg.inv(w), np.dot(Phi.T, t))

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
    # fig1.savefig('lin_reg.png')
    plt.show()

    fig2 = plt.figure()
    plt.stem(w, use_line_collection=True)
    plt.xlabel('basis')
    plt.ylabel('w')
    # fig2.savefig('lin_reg_w.png')
    plt.show()





