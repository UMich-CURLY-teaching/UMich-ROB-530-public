#!usr/bin/env python

#
# An example of pose, SE(3), synchronization. We build a pose graph where
# all factors are relative SE(3) measurements. Then we use Gauss-Newton to
# find a locally optimal solution.
#
# Author: Fangtong Liu
# Date: 05/17/2020
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import logm, expm
from lib import Rzyx


def skew(x):
    # vector to skew R^3 -> so(3)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def unskew(X):
    # so(3) -> R^3
    return np.array([X[2, 1], X[0, 2], X[1, 0]]).reshape(-1, 1)


def hat(x):
    # hat: R^6 -> se(3)
    matrix = np.hstack((skew(x[0:3]), x[3:6].reshape(-1, 1)))
    matrix = np.vstack((matrix, np.array([0, 0, 0, 0])))
    return matrix


def wedge(X):
    # wedge: se(3) -> R^6
    a = unskew(X[0:3, 0:3])
    matrix = np.vstack((unskew(X[0:3, 0:3]), X[0:3, 3].reshape(-1, 1)))
    return matrix


def Adjoint_SE3(X):
    # ADJOINT_SE3 Computes the adjoint of SE(3)
    line1 = np.hstack((X[0:3, 0:3], np.zeros([3, 3])))
    line2 = np.hstack((np.dot(skew(X[0:3, 3]), X[0:3, 0:3]), X[0:3, 0:3]))
    return np.vstack((line1, line2))


def LeftJacobian_SO3(w):
    # LEFT JACOBIAN as defined in http://perpustakaan.unitomo.ac.id/repository/Associating%20Uncertainty%20With%20Three-Dimensional.pdf
    theta = np.linalg.norm(w)
    A = skew(w)
    if theta == 0:
        return np.eye(3)
    Jacobian = np.eye(3) + (1 - np.cos(theta)) / (theta ** 2) * A + (theta - np.sin(theta)) / np.power(theta, 3) * np.dot(A, A)
    return Jacobian


def LeftJacobian_SE3(xi):
    # https://github.com/RossHartley/lie/blob/master/matlab/%2BLie/LeftJacobian_SE3.m

    Phi = xi[0:3]
    phi = np.linalg.norm(Phi)
    Rho = xi[3:6]
    Phi_skew = skew(Phi)
    Rho_skew = skew(Rho)
    J = LeftJacobian_SO3(Phi)

    if phi == 0:
        Q = 0.5 * Rho_skew
    else:
        Q = 0.5 * Rho_skew \
            + (phi - np.sin(phi)) / np.power(phi, 3) \
            * (np.dot(Phi_skew, Rho_skew) + np.dot(Rho_skew, Phi_skew) + np.dot(np.dot(Phi_skew, Rho_skew), Phi_skew)) \
            - (1 - 0.5 * np.power(phi, 2) - np.cos(phi)) / np.power(phi, 4) \
            * (np.dot(np.dot(Phi_skew, Phi_skew), Rho_skew) + np.dot(np.dot(Rho_skew, Phi_skew), Phi_skew)
               - 3 * np.dot(np.dot(Phi_skew, Rho_skew), Phi_skew)) \
            - 0.5 * ((1 - 0.5 * np.power(phi, 2) - np.cos(phi)) / np.power(phi, 4)
                     - 3 * (phi - np.sin(phi) - np.power(phi, 3) / 6) / np.power(phi, 5)) \
            * (np.dot(np.dot(np.dot(Phi_skew, Rho_skew), Phi_skew), Phi_skew)
               + np.dot(np.dot(np.dot(Phi_skew, Phi_skew), Rho_skew), Phi_skew))
    matrix = np.hstack((J, np.zeros([3, 3])))
    matrix = np.vstack((matrix, np.hstack((Q, J))))

    return matrix


def RightJacobian_SE3(xi):
    # RIGHT JACOBIAN as defined in http://perpustakaan.unitomo.ac.id/repository/Associating%20Uncertainty%20With%20Three-Dimensional.pdf
    return np.dot(Adjoint_SE3(expm(hat(-xi))), LeftJacobian_SE3(xi))


def RightJacobianInverse_SE3(xi):
    Jr = RightJacobian_SE3(xi)
    aa = np.linalg.inv(Jr)
    return np.linalg.inv(Jr)


def generator(x):
    matrix = np.hstack((skew(x), np.zeros([3, 1])))
    matrix = np.vstack((matrix, np.zeros([1, 4])))
    return matrix


def bodyTransformation(a, b, c, d, e, f):
    matrix = np.hstack((Rzyx(a, b, c), np.array([[d], [e], [f]])))
    matrix = np.vstack((matrix, np.array([0, 0, 0, 1])))
    return matrix


if __name__ == "__main__":

    # colors
    green = np.array([0.2980, 0.6, 0])
    crimson = np.array([220, 20, 60]) / 255
    darkblue = np.array([0, 0.2, 0.4])
    DupontGray = np.array([144, 131, 118]) / 255

    # se(3) generators; twist = vec(omega, v)
    # R^3 standard basis
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    # se(3) (SE(3) Lie algebra) infinitesimal generators
    G = []
    G.append(generator(e1))
    G.append(generator(e2))
    G.append(generator(e3))
    G.append(np.zeros([4, 4]))
    G.append(np.zeros([4, 4]))
    G.append(np.zeros([4, 4]))
    G[0][0, 3] = 1
    G[1][1, 3] = 1
    G[2][2, 3] = 1

    # a sequence of relative rigid body transformation
    U = []
    U.append(bodyTransformation(0.1, 0.01, -0.02, 0.8, 0.6, 0.9))  # T01
    U.append(bodyTransformation(0.1, -0.1, 0.2, 0.9, -0.6, 0.6))  # T12
    U.append(bodyTransformation(0.3, -0.1, -0.3, 1.2, 0.6, 1.0))  # T23
    U.append(bodyTransformation(-0.3, -0.1, -0.1, 1.2, 0.9, 0.8))  # T34
    U.append(bodyTransformation(-0.7, -0.1, -0.3, -1.4, 1.5, -1.1))  # T45

    # accumulate the transformation, starting from the identity
    Keys = []  # to track nodes in each factors
    T = []
    T.append(np.eye(4))
    for i in range(1, 6):
        T.append(np.dot(T[i-1], U[i-1]))
        Keys.append(np.array([i-1, i]))

    # create noisy SE(3) measurements
    # measurement noise covariance
    Sigma = np.diag(np.power([0.05, 0.03, 0.03, 0.05, 0.05, 0.05], 2))
    # Cholesky factor of covariance for sampling
    Lz = np.linalg.cholesky(Sigma)
    Z = []
    for i in range(5):
        # sample from a zero mean Gaussian
        noise = np.dot(Lz, np.random.randn(6, 1))
        # noise = np.ones([6, 1]) * 10
        N = np.zeros([4, 4])
        for j in range(len(G)):
            N = N + G[j] * noise[j]
        Z.append(np.dot(U[i], expm(N)))

    # add two loop-closures between T1-T5, and T3-T5
    l1 = [0, 3]
    l2 = [5, 5]
    for i in range(len(l1)):
        # sample from a zero mean Gaussian
        noise = np.dot(Lz, np.random.randn(6, 1))
        # noise = np.ones([6, 1]) * 10
        N = np.zeros([4, 4])
        for j in range(len(G)):
            N = N + G[j] * noise[j]
        Z.append(np.dot(np.dot(np.linalg.inv(T[l1[i]]), T[l2[i]]), expm(N)))
        Keys.append(np.array([l1[i], l2[i]]))

    # plot the pose graph
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    text_offset = [-0.1, -0.1, 0.5]
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$x_3$')
    plot_label = True
    for i in range(len(T)):
        if i == 0:
            ax.scatter3D(T[i][0, 3], T[i][1, 3], T[i][2, 3], 'o', color=crimson, alpha=0.5, s=15, label='Ground truth')
            ax.text(T[i][0, 3] + text_offset[0], T[i][1, 3] + text_offset[1], T[i][2, 3] + text_offset[2],
                    r'$T_{%d}$' % i, zdir='x')
        if i > 0:
            ax.scatter3D(T[i][0, 3], T[i][1, 3], T[i][2, 3], 'o', color=crimson, alpha=0.5, s=15)
            ax.text(T[i][0, 3] + text_offset[0], T[i][1, 3] + text_offset[1], T[i][2, 3] + text_offset[2],
                    r'$T_{%d}$' % i)
            x = np.array([T[i - 1][0, 3], T[i][0, 3]])
            y = np.array([T[i - 1][1, 3], T[i][1, 3]])
            z = np.array([T[i - 1][2, 3], T[i][2, 3]])
            ax.plot3D(x, y, z, '-', color=green, alpha=0.3, linewidth=4)
    # plot loop-closures
    k = 0
    for i in range(len(T)-1, len(Z)):
        x = np.array([T[l1[k]][0, 3], T[l2[k]][0, 3]])
        y = np.array([T[l1[k]][1, 3], T[l2[k]][1, 3]])
        z = np.array([T[l1[k]][2, 3], T[l2[k]][2, 3]])
        ax.plot3D(x, y, z, '-', color=green, alpha=0.3, linewidth=4)
        k = k + 1

    # compute initial guess by accumulating noisy measurements
    T_init = []
    T_init.append(np.eye(4))
    for i in range(1, 6):
        T_init.append(np.dot(T_init[i-1], Z[i-1]))

    # plot initial guess
    for i in range(len(T_init)):
        if i == 0:
            ax.scatter3D(T_init[i][0, 3], T_init[i][1, 3], T_init[i][2, 3], 's', color=DupontGray, alpha=0.5, s=15, label='Initial guess')
        if i > 0:
            ax.scatter3D(T_init[i][0, 3], T_init[i][1, 3], T_init[i][2, 3], 's', color=DupontGray, alpha=0.5, s=15)
            x = np.array([T_init[i - 1][0, 3], T_init[i][0, 3]])
            y = np.array([T_init[i - 1][1, 3], T_init[i][1, 3]])
            z = np.array([T_init[i - 1][2, 3], T_init[i][2, 3]])
            ax.plot3D(x, y, z, '-', color=DupontGray, alpha=0.3, linewidth=4)

    # We now construct the Jacobian matrix. The rows are the measurements which are
    # SE(3) here. Therefore, each measurement occupies 6 rows. Columns correspond to
    # the decision variables which are SE(3) here. Therefore, we have 6 x number of
    # poses variables. Note that the optimization is parametrized using twist (6 x 1)
    # which lives in the Lie algebra se(3). We find the correction twist and retract it
    # using Lie exp map and "add" (multiply) it to the previous iteration guess. This
    # process should be repeated for each pose before moving to the next iteration.
    # Further, we anchor the first pose to constrain the graph, i.e., we fix the first
    # pose to be at the identity. This will all an extra 6 rows to the Jacobian matrix
    # and is equivalent of placing a prior over the first node of the graph

    # Jacobian matrix
    A = np.zeros([6 + 6 * len(Z), 6 * len(T_init)])
    # right hand side (residuals)
    b = np.zeros([6 + 6 * len(Z), 1])
    # anchor node covariance; we want to fix the node so the covariance should be small.
    # This will result in large weights in the optimization process.
    Sigma_init = np.eye(6) * np.power(0.1, 2)
    A[0:6, 0:6] = np.linalg.inv(np.linalg.cholesky(Sigma_init))

    # Gauss-Newton solver over SE(3)
    T_est = np.copy(T_init)
    max_iter = 100
    iter = 0
    eps_Jr = 1e-9
    dx = np.zeros([6 + 6 * len(Z), 1])
    r = np.zeros([6 + 6 * len(Z), 1])
    # SE(3) right Jacobian inverse and adjoint
    invJr = RightJacobianInverse_SE3
    Ad = Adjoint_SE3
    while iter < max_iter:
        iter += 1
        # compute residual
        r[0:dx.shape[0], 0] = np.copy(dx).reshape(-1)
        r[0:6, 0] = wedge(logm(T_est[0])).reshape(1, -1)
        for i in range(len(Keys)):
            key = Keys[i]
            res_idx = np.arange(6 * (i+1), 6 * (i+1) + 6, 1)
            r[res_idx, 0] = wedge(logm(np.dot(np.linalg.inv(Z[i]), np.dot(np.linalg.inv(T_est[key[0]]),
                                                                          T_est[key[1]])))).reshape(-1)
            # fill in Jacobian of the corresponding target position
            idx = 6 * key[0]
            A[res_idx, idx:idx+6] = np.dot(np.dot(np.linalg.inv(Lz), -invJr(r[res_idx, 0])),
                                           Ad(np.dot(np.linalg.inv(T_est[key[1]]), T_est[key[0]])))
            idx = 6 * key[1]
            A[res_idx, idx:idx+6] = np.dot(np.linalg.inv(Lz), invJr(r[res_idx, 0]))
            r[res_idx, 0] = np.dot(np.linalg.inv(Lz), r[res_idx, 0])

        # solve normal equations
        Jr = np.dot(-A.T, r)
        dx = np.dot(np.linalg.inv(np.dot(A.T, A)), Jr)
        print('iter: ', iter)
        # retract and update the estimate
        for i in range(len(T_init)):
            xi = dx[6 * i:6 * (i + 1)]
            T_est[i] = np.dot(T_est[i], expm(hat(xi)))
        # check if converged
        if np.linalg.norm(Jr) < eps_Jr:
            break

    # plot result
    for i in range(len(T_est)):
        if i == 0:
            ax.scatter3D(T_est[i][0, 3], T_est[i][1, 3], T_est[i][2, 3], '*', color=darkblue, alpha=0.5, s=15, label='Gauss-Newton')
        if i > 0:
            ax.scatter3D(T_est[i][0, 3], T_est[i][1, 3], T_est[i][2, 3], '*', color=darkblue, alpha=0.5, s=15)
            x = np.array([T_est[i - 1][0, 3], T_est[i][0, 3]])
            y = np.array([T_est[i - 1][1, 3], T_est[i][1, 3]])
            z = np.array([T_est[i - 1][2, 3], T_est[i][2, 3]])
            ax.plot3D(x, y, z, '-', color=darkblue, alpha=0.3, linewidth=4)

    ax.view_init(elev=25., azim=-150)
    plt.legend(loc='best')
    plt.show()








