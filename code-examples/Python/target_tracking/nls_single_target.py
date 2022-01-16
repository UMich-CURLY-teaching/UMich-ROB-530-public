#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 04/04/2020
#

import numpy as np
from numpy.random import randn
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix, linalg
import matplotlib.pyplot as plt
import math


class myStruct:
    pass

# measurement model
def measurement_model(x):
    h = np.array([[np.sqrt(np.sum(x[0] ** 2 + x[1] ** 2))],
                  [math.atan2(x[0], x[1])]])
    return h.reshape([2, 1])


# measurement model Jacobian
def measurement_Jacobian(x):
    H = np.array([[x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2), x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2)],
                  [x[1] / (x[0] ** 2 + x[1] ** 2), -x[0] / (x[0] ** 2 + x[1] ** 2)]])
    return H.reshape([2, 2])


# Toy example for tracking a single target using batch nonlinear least squares
if __name__ == '__main__':
    # First simulate a target that moves on a curved path; we assume ownship is at the origin (0, 0)
    # and received noisy range and bearing measurement of the target location. There is no knowledge
    # of the target motion, but we assume target is close to its previous location to constrain the state

    # ground truth data
    gt = myStruct()
    gt.x = np.arange(-5, 5.1, 0.1)
    gt.y = np.array(1 * np.sin(gt.x) + 3)

    # measurements
    R = np.diag(np.power([0.1, 0.05], 2))
    # Cholesky factor of covariance for sampling
    Lz = np.linalg.cholesky(R)

    z = np.zeros([2, len(gt.x)])
    for i in range(len(gt.x)):
        # sample from a zero mean Gaussian with covariance R
        noise = np.dot(Lz, randn(2, 1)).reshape(-1)
        z[:, i] = np.array([np.sqrt(gt.x[i] ** 2 + gt.y[i] ** 2), math.atan2(gt.x[i], gt.y[i])]) + noise

    # Jacobians of the motion model (we assume the transition model is identity,
    # this will enforce the adjacent points remain close)
    F = -np.eye(2)
    G = np.eye(2)
    Sigma_u = np.diag(np.power([0.05, 0.05], 2))
    Lu = np.linalg.cholesky(Sigma_u)

    # Jacobians of measurement model
    h = measurement_model
    H = measurement_Jacobian

    # initial guess; in general for a nonlinear problem finding a good initial guess can be difficult.
    # Local solvers are sensitive to initial guess and only find local minimum. Here using each range and
    # bearing measurement, the target position is fully observable so we can initialize using given
    # noisy measurements.
    x_init = []
    for i in range(z.shape[1]):
        x_init.append(z[0, i] * np.sin(z[1, i]))
        x_init.append(z[0, i] * np.cos(z[1, i]))
    x_init = np.array(x_init)

    # constructing A using target motion
    nt = 2  # dimension of the target position
    Sigma_init = np.diag(np.power([0.5, 0.5], 2))
    A = np.dot(np.linalg.inv(np.linalg.cholesky(Sigma_init)), G)  # apply noise covariance as weight
    for i in range(1, np.shape(z)[1], 1):
        A = block_diag(A, np.dot(np.linalg.inv(Lu), G))
        A[-nt:, -2 * nt:-nt] = np.dot(np.linalg.inv(Lu), F)

    # Compute b (right hand side) for linear constraints
    b = []
    for i in range(np.shape(z)[1]):
        temp = np.dot(np.linalg.inv(Lu), np.zeros([2, 1])).reshape(-1)
        b.append(temp)

    # filling A using sensor measurements
    for i in range(np.shape(z)[1]):
        [nA, mA] = np.shape(A)
        Asub = np.zeros([nt, mA])  # new submatrix in the bottom of A
        # fill in Jacobian of the corresponding target position
        Asub[:, (2 * i):(2 * i + 2)] = np.dot(np.linalg.inv(Lz), H(x_init[2*i:2*i+2]))
        # append A and b
        A = np.vstack((A, Asub))
        # b.append(np.dot(np.linalg.inv(Lz), z[:, i]).reshape(-1))
    b = np.array(b).reshape(-1)
    nb = len(b)

    A = csr_matrix(A)

    # Gauss-Newton solver
    x_target = np.copy(x_init)
    max_iter = 1000
    iter = 0
    eps_Jr = 1e-6
    while iter < max_iter:
        iter += 1
        # compute residual
        r = []
        b = x_target[0:2]

        for i in range(z.shape[1]):
            r.append(np.dot(np.linalg.inv(Lz), (z[:, i].reshape(2, 1) - h(x_target[2*i:2*i+2]))).reshape(-1))
            if i < z.shape[1]-1:
                b = np.hstack((b, np.dot(np.linalg.inv(Lu), (x_target[2*i+2:2*i+4] - x_target[2*i:2*i+2])).reshape(-1)))
            # fill in Jacobian of the corresponding target position
            idx = nb + 2 * i
            A[idx:idx+2, 2*i:2*i+2] = np.dot(np.linalg.inv(Lz), -H(x_target[2*i:2*i+2]))
        r = np.array(r).reshape(-1)
        r = np.hstack((b, r)).reshape(-1, 1)

        # solve normal equations

        Jr = - A.T @ r

        dx = linalg.inv(A.T @ A) @ Jr
        x_target = x_target + dx.reshape(-1)

        # check if converged
        if np.linalg.norm(Jr) < eps_Jr:
            break


    # plotting
    green = np.array([0.2980, 0.6, 0])
    darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    VermillionRed = np.array([156, 31, 46]) / 255

    fig = plt.figure()
    line1, = plt.plot(0, 0, '^', color='b', markersize=20)
    line2, = plt.plot(x_init[0::2], x_init[1::2], '-', linewidth=2, alpha=0.8, color=green)
    line3, = plt.plot(gt.x, gt.y, '-', color=Darkgrey, alpha=1, linewidth=2)
    line4, = plt.plot(x_target[0::2], x_target[1::2], '--', linewidth=1.5, alpha=0.8, color=VermillionRed)
    plt.legend([line1, line2, line3, line4], [r'ownship', r'initial geuss', r'ground truth', r'Gauss-Newton'], loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()

