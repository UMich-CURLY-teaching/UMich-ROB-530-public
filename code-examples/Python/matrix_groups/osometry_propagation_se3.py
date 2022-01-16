#!usr/bin/env python

#
# An example of process models on Lie groups and uncertainty propagation
# using SE(3). The process model is simpley X_k+1 = X_k * exp(U_k + w_k) where
# X_k is in SE(3) and U_k and w_k (zero mean white Gaussian noise, N(0, Q_k))
# are define in the Lie algebra se(3). We use Monte Carlo methods to propagate
# samples over a path and then compute the sample mean and covariance on Lie group,
# here SE(3). Note that the sample mean and covariance are computed using an
# iterative algorithm which is different than usual Euclidean sample statistics.
# The covariance on Lie algebra is flat as expected but it's nonlinear when mapped
# to the manifold using Lie exp map.
#
# Author: Fangtong Liu
# Date: 05/17/2020
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import logm, expm
from lib import Rzyx


def pos_def_matrix(A):
    # Check input matrix is positive definite or not
    return np.all(np.linalg.eigvals(A) > 0)    

def adjoint(X):
    # SE(3) Adjoint
    matrix = np.hstack((X[0:3, 0:3], np.dot(skew(X[0:3, 3]), X[0:3, 0:3])))
    matrix = np.vstack((matrix, np.hstack((np.zeros([3, 3]), X[0:3, 0:3]))))
    return matrix

def skew(x):
    # vector to skew R^3 -> so(3)
    matrix = np.array([[0, -x[2], x[1]],
                       [x[2], 0, -x[0]],
                       [-x[1], x[0], 0]])
    return matrix


def unskew(A):
    # so(3) -> R^3
    return np.array([[A[2, 1]], [A[0, 2]], [A[1, 0]]])


def hat(x):
    # hat: R^6 -> se(3)
    matrix = np.hstack((skew(x[3:6]), x[0:3].reshape(-1, 1)))
    matrix = np.vstack((matrix, np.array([0, 0, 0, 0])))
    return matrix


def wedge(X):
    # wedge: se(3) -> R^6
    matrix = np.vstack((X[0:3, 3].reshape(-1, 1), unskew(X[0:3, 0:3])))
    return matrix


def propagation(robot, u):
    # SE(3) propagation model; each input is U \in se(3) plus exp map of the
    # noise define in LIe algebra
    Ui = u[0] * robot['G1'] + u[1] * robot['G2'] + u[2] * robot['G3'] \
         + u[3] * robot['G4'] + u[4] * robot['G5'] + u[5] * robot['G6']
    for i in range(robot['n']):
        # sample from a zero mean Gausian
        noise = np.dot(robot['L'], np.random.randn(6, 1))
        N = noise[0] * robot['G1'] + noise[1] * robot['G2'] + noise[2] * robot['G3'] \
            + noise[3] * robot['G4'] + noise[4] * robot['G5'] + noise[5] * robot['G6']
        robot['x'][i] = np.dot(robot['x'][i], expm(Ui + N))
    return robot


def Lie_sample_statistics(robot):
    # compute sample mean and covariance on matrix Lie group
    mu0 = robot['x'][0]  # pick a sample as initial guess
    v = np.copy(robot['x'])
    max_iter = 100
    iter = 1
    while iter < max_iter:
        mu = mu0 * 0
        Sigma = np.zeros([6, 1])
        for i in range(robot['n']):
            # left-invariant error: eta^L = X^-1 * X^hat
            v[i] = logm(np.dot(np.linalg.inv(mu0), robot['x'][i]))
            mu = mu + v[i]
            vec_v = wedge(v[i])
            Sigma = Sigma + np.dot(vec_v, vec_v.T)
        mu = np.dot(mu0, expm(mu / robot['n']))
        Sigma = (1 / (robot['n'] - 1)) * Sigma  # unbiased sample covariance
        # check if we are done here
        temp = np.linalg.norm(logm(np.dot(np.linalg.inv(mu0), mu)))
        if temp < 1e-8:
            return mu, Sigma
        else:
            mu0 = np.copy(mu)
        iter += 1
    print('\033[91mWarning: Not converged! Max iteration reached. The statistic might not be reliable.\033[0m')


if __name__ == "__main__":

    # se(3) generators; twist = vec(v, omega).
    # R^3 standard basis
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    # so(3) (SO(3) Lie algebra) infinitesimal generator
    robot = {}
    robot['G1'] = np.zeros([4, 4])
    robot['G1'][0, 3] = 1

    robot['G2'] = np.zeros([4, 4])
    robot['G2'][1, 3] = 1

    robot['G3'] = np.zeros([4, 4])
    robot['G3'][2, 3] = 1

    robot['G4'] = np.hstack((skew(e1), np.zeros([3, 1])))
    robot['G4'] = np.vstack((robot['G4'], np.zeros([1, 4])))

    robot['G5'] = np.hstack((skew(e2), np.zeros([3, 1])))
    robot['G5'] = np.vstack((robot['G5'], np.zeros([1, 4])))

    robot['G6'] = np.hstack((skew(e3), np.zeros([3, 1])))
    robot['G6'] = np.vstack((robot['G6'], np.zeros([1, 4])))

    # SE(3) Adjoint
    robot['Ad'] = adjoint

    # generate noise-free control inputs in Lie algebra se(3)
    nstep = 4
    u = []
    u.append(np.linspace(0, 4, nstep))
    u.append(1.5 * u[0])
    u.append(0.05 * np.exp(0.6 * u[0]) - 0.05)
    u.append(np.linspace(0, 0.5, nstep))
    u.append(np.linspace(0, 0.3, nstep))
    u.append(np.linspace(0, 0.7, nstep))
    u = np.array(u)

    # build a 3D robot!
    robot['n'] = 1000
    robot['x'] = []  # state mean
    robot['Cov_fo'] = np.zeros([6, 6])  # first order covariance propagation around mean
    for i in range(robot['n']):
        robot['x'].append(np.eye(4))
    # motion model noise covariance
    robot['Q'] = np.diag(np.power([0.01, 0.01, 0.05, 0.05, 0.05, 0.2], 2))
    # Cholesky factor of covariance for sampling
    robot['L'] = np.linalg.cholesky(robot['Q'])

    # construct noise free motion trajectory
    path = {}
    path['T'] = np.eye(4)
    path['x'] = [0]
    path['y'] = [0]
    path['z'] = [0]
    for i in range(u.shape[1]):
        Ui = u[0, i] * robot['G1'] + u[1, i] * robot['G2'] + u[2, i] * robot['G3'] \
             + u[3, i] * robot['G4'] + u[4, i] * robot['G5'] + u[5, i] * robot['G6']
        path['T'] = np.dot(path['T'], expm(Ui))
        path['x'].append(path['T'][0, 3])
        path['y'].append(path['T'][1, 3])
        path['z'].append(path['T'][2, 3])

    # create confidence ellipsoid
    # first create points from a unit sphere
    phi = np.linspace(-np.pi, np.pi, 100)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
    PHI, THETA = np.meshgrid(phi, theta)
    X_sph = np.cos(THETA) * np.cos(PHI)
    Y_sph = np.sin(THETA) * np.sin(PHI)
    Z_sph = np.sin(THETA)
    sphere = np.array([X_sph.reshape(-1), Y_sph.reshape(-1), Z_sph.reshape(-1)])
    sphere = np.vstack((sphere, np.zeros(sphere.shape))).T
    # Chi-squared 6-DOF 95% confidence (0.05): 12.592
    scale = np.sqrt(12.592)

    # incremental visualization
    green = np.array([0.2980, 0.6, 0])
    crimson = np.array([220, 20, 60]) / 255
    darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    VermillionRed = np.array([156, 31, 46]) / 255
    DuponGray = np.array([144, 131, 118]) / 255

    # plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    ax.plot3D(path['x'], path['y'], path['z'], '-', color=Darkgrey, linewidth=3, label='Normal path')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$x_3$')

    # extract propagated particles
    p = np.zeros([3, robot['n']])
    for i in range(robot['n']):
        p[0, i] = robot['x'][i][0, 3]
        p[1, i] = robot['x'][i][1, 3]
        p[2, i] = robot['x'][i][2, 3]

    # compute sample statistics
    mu, Sigma = Lie_sample_statistics(robot)

    # plot sample mean and particles
    ax.scatter3D(p[0, :], p[1, :], p[2, :], '.', color=green, alpha=0.5, s=1, label='Samples')
    ax.scatter3D(mu[0, 3], mu[1, 3], mu[2, 3], 'o', color=crimson, s=1.5, label='Sample mean')

    # main loop; iterate over the control inputs and move the robot
    ELLIPSOID = np.zeros([sphere.shape[0], 3])  # covariance ellipse on manifold (nonlinear)
    ELLIPSOID_fo = np.zeros([sphere.shape[0], 3])  # first order covariance ellipse on manifold (nonlinear)
    ellipsoid = np.zeros([sphere.shape[0], 3])  # covariance ellipse on Lie algebra
    plot_label = True
    for i in range(u.shape[1]):
        # move particles based on input
        robot = propagation(robot, u[:, i])
        # extract propagated particles
        p = np.zeros([3, robot['n']])
        for j in range(robot['n']):
            p[0, j] = robot['x'][j][0, 3]
            p[1, j] = robot['x'][j][1, 3]
            p[2, j] = robot['x'][j][2, 3]

        # show particles
        ax.scatter3D(p[0, :], p[1, :], p[2, :], '.', color=green, alpha=0.5, s=1)

        # compute sample statistics
        mu, Sigma = Lie_sample_statistics(robot)

        # compute first order analytical covariance propagation
        Ui = expm(u[0, i] * robot['G1'] + u[1, i] * robot['G2'] + u[2, i] * robot['G3']
                  + u[3, i] * robot['G4'] + u[4, i] * robot['G5'] + u[5, i] * robot['G6'])

        # left-invariant error: eta^L = X^-1 * X^hat
        # robot.Ad(inv(Ui)) maps the covariance back to Lie algebra using the
        # incremental motion Ui (hence inv(Ui)). Then the noise covariance that
        # is already defined in Lie algebra can be added to the mapped state covariance
        temp_value = robot['Ad'](np.linalg.inv(Ui))
        robot['Cov_fo'] = np.dot(np.dot(temp_value, robot['Cov_fo']), temp_value.T) + robot['Q']
        robot['L_fo'] = np.linalg.cholesky(robot['Cov_fo'])

        # create the ellipse using the unit circle
        if pos_def_matrix(Sigma):
            for j in range(sphere.shape[0]):
                L = np.linalg.cholesky(Sigma)
                # sample covariance on SE(2)
                ell_se3_vec = scale * np.dot(L, sphere[j, :])
                # retract and left-translate the ellipse on LIe algebra to SE(3) using Lie exp map
                temp = np.dot(mu, expm(robot['G1'] * ell_se3_vec[0] + robot['G2'] * ell_se3_vec[1]
                                       + robot['G3'] * ell_se3_vec[2] + robot['G4'] * ell_se3_vec[3]
                                       + robot['G5'] * ell_se3_vec[4] + robot['G6'] * ell_se3_vec[5]))
                ELLIPSOID[j, :] = np.array([temp[0, 3], temp[1, 3], temp[2, 3]])

                # get the ellipse on Lie algebra
                temp_matrix = np.hstack(([Rzyx(ell_se3_vec[3], ell_se3_vec[4], ell_se3_vec[5]), ell_se3_vec[0:3].reshape(-1, 1)]))
                temp_matrix = np.vstack((temp_matrix, np.array([0, 0, 0, 1])))
                temp = np.dot(mu, temp_matrix)
                ellipsoid[j, :] = np.array([temp[0, 3], temp[1, 3], temp[2, 3]])

                # sample covariance on SE(3)
                ell_se3_vec = scale * np.dot(robot['L_fo'], sphere[j, :])
                # retract and left-translate the ellipse on Lie algebra to SE(3) using Lie exp map
                temp = np.dot(mu, expm(robot['G1'] * ell_se3_vec[0] + robot['G2'] * ell_se3_vec[1]
                                       + robot['G3'] * ell_se3_vec[2] + robot['G4'] * ell_se3_vec[3]
                                       + robot['G5'] * ell_se3_vec[4] + robot['G6'] * ell_se3_vec[5]))
                ELLIPSOID_fo[j, :] = np.array([temp[0, 3], temp[1, 3], temp[2, 3]])

            # plot the ellipsoid
            # extract x, y, z matrices for plotting
            X_ell = ELLIPSOID[:, 0].reshape(X_sph.shape)
            Y_ell = ELLIPSOID[:, 1].reshape(Y_sph.shape)
            Z_ell = ELLIPSOID[:, 2].reshape(Z_sph.shape)
            if plot_label:
                surf1 = ax.plot_surface(X_ell, Y_ell, Z_ell, color=VermillionRed, alpha=0.2,
                                       label='Sample covariance - SE(3)')
                surf1._facecolors2d = surf1._facecolors3d
                surf1._edgecolors2d = surf1._edgecolors3d
            else:
                surf1 = ax.plot_surface(X_ell, Y_ell, Z_ell, color=VermillionRed, alpha=0.2)
                surf1._facecolors2d = surf1._facecolors3d
                surf1._edgecolors2d = surf1._edgecolors3d

            # extract x, y, z matrices for plotting
            X_ell = ELLIPSOID_fo[:, 0].reshape(X_sph.shape)
            Y_ell = ELLIPSOID_fo[:, 1].reshape(Y_sph.shape)
            Z_ell = ELLIPSOID_fo[:, 2].reshape(Z_sph.shape)
            if plot_label:
                surf2 = ax.plot_surface(X_ell, Y_ell, Z_ell, color=darkblue, alpha=0.2,
                                        label='First-order covariance - SE(3)')
                surf2._facecolors2d = surf2._facecolors3d
                surf2._edgecolors2d = surf2._edgecolors3d
            else:
                surf2 = ax.plot_surface(X_ell, Y_ell, Z_ell, color=darkblue, alpha=0.2)
                surf2._facecolors2d = surf2._facecolors3d
                surf2._edgecolors2d = surf2._edgecolors3d

            # extract x, y, z matrices for plotting
            X_ell = ellipsoid[:, 0].reshape(X_sph.shape)
            Y_ell = ellipsoid[:, 1].reshape(Y_sph.shape)
            Z_ell = ellipsoid[:, 2].reshape(Z_sph.shape)
            if plot_label:
                surf3 = ax.plot_surface(X_ell, Y_ell, Z_ell, color=DuponGray, alpha=0.2,
                                        label='Sample covariance - Lie algebra')
                surf3._facecolors2d = surf3._facecolors3d
                surf3._edgecolors2d = surf3._edgecolors3d
                plot_label = False
            else:
                ax.plot_surface(X_ell, Y_ell, Z_ell, color=DuponGray, alpha=0.2)
                surf3._facecolors2d = surf3._facecolors3d
                surf3._edgecolors2d = surf3._edgecolors3d

        ax.scatter3D(mu[0, 3], mu[1, 3], mu[2, 3], 'o', color=crimson, alpha=0.7, s=1.5)
        plt.legend(loc='upper left')
        plt.pause(0.05)
    fig.savefig('banana_is_gaussian_se3.png')
    plt.show()


