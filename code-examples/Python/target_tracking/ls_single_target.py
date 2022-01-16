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


class myStruct:
    pass

# Toy example for tracking a single target using batch least squares
if __name__ == '__main__':
    # First simulate a target that moves on a curved path; we assume ownship is at the origin (0, 0)
    # and received noisy (x, y) measurements of the target location. There is no knowledge
    # of the target motion, but we assume target is close to its previous location to constrain the state

    # ground truth data
    gt = myStruct()
    gt.x = np.arange(-5, 5.1, 0.1)
    gt.y = np.array(1 * np.sin(gt.x) + 3)

    # measurements
    R = np.diag(np.power([0.05, 0.05], 2))
    # Cholesky factor of covariance for sampling
    Lz = np.linalg.cholesky(R)
    z = np.zeros([2, len(gt.x)])
    for i in range(len(gt.x)):
        # sample from a zero mean Gaussian with covariance R
        noise = np.dot(Lz, randn(2, 1))
        # noise = np.dot(L, np.array([[0.05], [0.1]])).reshape(-1)
        z[:, i] = (np.array([[gt.x[i]], [gt.y[i]]]) + noise).reshape(-1)

    # Jacobians of the motion model (we assume the transition model is identity,
    # this will enforce the adjacent points remain close)
    F = -np.eye(2)
    G = np.eye(2)
    Sigma_u = np.diag(np.power([0.03, 0.03], 2))
    Lu = np.linalg.cholesky(Sigma_u)

    # Jacobians of measurement model
    H = np.eye(2)

    # constructing A using target motion
    nt = 2  # dimension of the target position
    Sigma_init = np.diag(np.power([0.8, 0.8], 2))
    A = np.dot(np.linalg.inv(np.linalg.cholesky(Sigma_init)), G)  # apply noise covariance as weight

    for i in range(1, np.shape(z)[1], 1):
        A = block_diag(A, np.dot(np.linalg.inv(Lu), G))
        A[-nt:, -2*nt:-nt] = np.dot(np.linalg.inv(Lu), F)

    # Compute b (right hand side)
    b = []
    for i in range(np.shape(z)[1]):
        temp = np.dot(np.linalg.inv(Lu), np.zeros([2, 1])).reshape(-1)
        b.append(temp)

    # filling A using sensor measurements
    for i in range(np.shape(z)[1]):
        [nA, mA] = np.shape(A)
        Asub = np.zeros([nt, mA])  # new submatrix in the bottom of A
        # fill in Jacobian of the corresponding target position
        Asub[:, (2*i):(2*i+2)] = np.dot(np.linalg.inv(Lz), H)
        # append A and b
        A = np.vstack((A, Asub))
        b.append(np.dot(np.linalg.inv(Lz), z[:, i]).reshape(-1))
    b = np.array(b).reshape(-1)
    
    A = csr_matrix(A)

    # Solve Ax=b for the target trajectory
    Ainv = linalg.inv(A.T @ A) @ A.T
    x_target = Ainv @ b

    # Solve using QR favtorization
    # x_qr = sparse.linalg.lsqr(A, b)
    [Q, R] = np.linalg.qr(A.toarray())
    x_qr = np.dot(np.dot(np.linalg.inv(R), Q.T), b)


    # Solve using Cholesky factorization
    L = np.linalg.cholesky((A.T @ A).toarray())
    x_chol = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L), A.T @ b))

    print(r'norm(A \ b - R \ Q^T * b) = ', np.linalg.norm(x_target - x_qr))
    print(r'norm(A \ b - L^T \ (L \ A^T * b)) = ', np.linalg.norm(x_target - x_chol))

    # plotting
    darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    VermillionRed = np.array([156, 31, 46]) / 255

    fig1 = plt.figure()
    line1, = plt.plot(0, 0,'^', color='b', markersize=20)
    line2, = plt.plot(gt.x, gt.y, '-', linewidth=2)
    line3, = plt.plot(x_target[0::2], x_target[1::2], '--r', linewidth=1.5)
    plt.legend([line1, line2, line3], [r'ownship', r'ground truth', r'Least Squares'], loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    # plt.show()

    fig2 = plt.figure(figsize=(12, 12))

    plt.subplot(221)
    ax1 = plt.gca()
    ax1.set_title(r'$A$')
    ax1.spy(A, markersize=.5, color=Darkgrey, origin='lower', aspect='equal')
    # h = plt.gca().get_children()

    plt.subplot(222)
    ax2 = plt.gca()
    ax2.set_title(r'$A^\mathsf{T}A$')
    ax2.spy(A.T @ A, markersize=.5, color=Darkgrey)
    # h = plt.gca().get_children()

    plt.subplot(223)
    ax3 = plt.gca()
    ax3.set_title(r'$\mathsf{R}-QR$')
    ax3.spy(R[0:np.shape(A.toarray())[1], :], markersize=.5, color=darkblue)
    # h = plt.gca().get_children()

    plt.subplot(224)
    ax4 = plt.gca()
    ax4.set_title(r'$\mathsf{R}-Cholesky$')
    ax4.spy(L.T, markersize=.5, color=VermillionRed)
    # h = plt.gca().get_children()
    plt.show()


