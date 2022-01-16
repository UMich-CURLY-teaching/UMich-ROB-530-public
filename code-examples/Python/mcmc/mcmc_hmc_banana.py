#!usr/bin/env python

#
# MCMC (Hamiltonian Monte Carlo) example for sampling from a banana distribution
# http://www.stat.columbia.edu/~gelman/research/published/GelmanMeng1991
# Author: Fangtong Liu
# Date: 06/14/2020
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def leapfrog_integration(z, r, J):
    e = 0.1
    r_half = r - e / 2 * J(z[0], z[1])
    z_new = z + e * r_half
    r_new = r_half - e / 2 * J(z_new[0], z_new[1])
    return z_new, r_new


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

    # HMC
    # We construct the Hamintonian; The total energy of the system is the sum of its potential,
    # E(z) = -log(f(z(0), z(1))), and kinetic, K(r) = 0.5 * r.T * R, energies
    def H(z, r):
        return -np.log(f(z[0], z[1])) + 0.5 * np.dot(r.T, r).reshape(-1)


    # We need the Jacobian of E for the leapfrog integration
    # E = (A * np.power(x1, 2) * np.power(x2, 2) + np.power(x1, 2) + np.power(x2, 2)
    #                                - 2 * B * x1 * x2 - 2 * C1 * x1 - 2 * C2 * x2))
    # J = jacobian(E,[x1, x2]).T;
    def J(x1, x2):

        jacobian = np.vstack((2 * A * np.dot(x1, np.power(x2, 2)) - 2 * B * x2 - 2 * C1 + 2 * x1,
                              2 * A * np.dot(x2, np.power(x1, 2)) - 2 * B * x1 - 2 * C2 + 2 * x2))
        return jacobian


    rejected = 0
    accepted = 0
    Max_iter = 10000

    # proposal distribution
    r = np.zeros([2, 1])
    prop_cov = np.diag([1, 1])
    prop_L = np.linalg.cholesky(prop_cov)
    x = np.zeros([2, 1])
    sample = []

    for i in range(Max_iter):
        # draw a random momentom from the proposal distribution
        r_new = r + np.dot(prop_L, np.random.randn(2, 1))
        # print(np.random.randn(2, 1))
        # take a leapfrog step
        x_new, r_new = leapfrog_integration(x, r_new, J)

        # accept with probability A
        u = np.random.rand(1)
        # print(i, np.log(f(x[0], x[1])), np.log(f(x_new[0], x_new[1])))
        A = min(1, np.exp(H(x, r) - H(x_new, r_new)))

        if A > u:  # accept
            x = np.copy(x_new)
            r = np.copy(r_new)
            sample.append(x)
            accepted = accepted + 1
        else:
            rejected = rejected + 1
    sample = np.array(sample).reshape(-1, 2)
    # print(sample)

    # plot
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(111, projection='3d')
    # hist, xedges, yedges = np.histogram2d(sample[:, 0], sample[:, 1], bins=(100, 100), range=[[-1, 6], [-1, 6]])
    # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    # xpos = xpos.flatten() / 2.
    # ypos = ypos.flatten() / 2.
    # zpos = np.zeros_like(xpos)
    #
    # dx = xedges[1] - xedges[0]
    # dy = yedges[1] - yedges[0]
    # dz = hist.flatten()
    #
    # cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    # max_height = np.max(dz)  # get range of colorbars so we can normalize
    # min_height = np.min(dz)
    # # scale each z to [0,1], and get their rgb values
    # rgba = [cmap((k - min_height) / max_height) for k in dz]
    #
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    # plt.hist2d(sample[:, 0], sample[:, 1], bins=100)
    # plt.xlim([-1, 6])
    # plt.ylim([-1, 6])
    # plt.show()



