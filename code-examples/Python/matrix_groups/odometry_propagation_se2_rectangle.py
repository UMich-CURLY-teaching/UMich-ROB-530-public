#!usr/bin/env python

#
# An example of velocity-based motion model on SE(2). The process model
# is simply X_k+1 = X_k * exp(u_k + w_k) where X_k is in SE(2), u_k is the
# twist in se(2), and w_k is N(0,Q_k) and defined in the Lie algebra se(2).
# We use Monte Carlo methods to propagate samples over a path and then compute
# the sample mean and covariance. Note that the sample mean and covariance are
# computed using an iterative algorithm which is different than usual Euclidean
# sample statistics.
# The covariance on Lie algebra is flat as expected but it's nonlinear when
# mapped to the manifold using Lie exp map. We use expm and logm as
# numerical Lie exp and log map. Both maps have closed-form formulas as
# well.
#
# Author: Fangtong Liu
# Date: 05/15/2020
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm


def posemat(x, y, h):
    # construct a SE(2) matrix element
    H = np.array([[np.cos(h), -np.sin(h), x],
                  [np.sin(h), np.cos(h), y],
                  [0, 0, 1]])
    return H

def adjoint(X):
    # SE(2) Adjoint
    matrix = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
    matrix = np.vstack((matrix, np.array([0, 0, 1])))
    return matrix


def propagation(robot, u):
    # SE(2) propagation model; the input is u \in se(2) plus noise
    # sample from a zero mean Gaussian
    noise = np.dot(robot['L'], np.random.randn(3, 1))
    N = robot['G1'] * noise[0] + robot['G2'] * noise[1] + robot['G3'] * noise[2]
    xi = u + N
    # propagate mean
    robot['x'] = np.dot(robot['x'], expm(xi))
    # propagate covariance
    robot['Cov'] = np.dot(np.dot(robot['Ad'](expm(-xi)), robot['Cov']), robot['Ad'](expm(-xi)).T) + robot['Q']
    return robot


if __name__ == "__main__":

    # generate a path
    dt = 1
    gt = {}
    n = len(np.arange(0, 10+dt, dt))
    gt['x'] = np.hstack((np.arange(0, 10+dt, dt).reshape(1, -1), 10 * np.ones([1, n]),
                         np.arange(10, 0-dt, -dt).reshape(1, -1), np.zeros([1, n])))
    gt['y'] = np.hstack((np.zeros([1, n]), np.arange(0, 10+dt, dt).reshape(1, -1),
                         10 * np.ones([1, n]), np.arange(10, 0-dt, -dt).reshape(1, -1)))

    # find the headings tangent to the path
    gt['h'] = []
    gt['h'].append(0)
    for i in range(1, gt['x'].shape[1]):
        gt['h'].append(np.arctan2(gt['y'][0, i] - gt['y'][0, i-1], gt['x'][0, i] - gt['x'][0, i-1]))
    # ground truth poses
    H = [np.eye(3)]
    for i in range(1, gt['x'].shape[1]):
        H.append(posemat(gt['x'][0, i], gt['y'][0, i], gt['h'][i]))
    # generate noise=free twist control inputs (velocity commands) in the Lie algebra
    u = [np.zeros([3, 3])]
    for i in range(1, gt['x'].shape[1]):
        logm_temp = logm(np.dot(np.linalg.inv(H[i-1]), H[i]))
        logm_temp = np.array(logm_temp)
        for m in range(logm_temp.shape[0]):
            for n in range(logm_temp.shape[1]):
                if np.iscomplex(logm_temp[m, n]):
                    logm_temp = np.imag(logm_temp)
                    temp = np.copy(logm_temp[1, :])
                    logm_temp[1, :] = logm_temp[0, :]
                    logm_temp[0, :] = -temp.reshape(1, -1)
                    pass
        u.append(logm_temp)

    # construct noise free motion trajectory (sanity check for the generated inputs)
    path = {}
    path['T'] = H[0]
    path['x'] = [0, 0]
    path['y'] = [0, 0]
    for i in range(1, len(u)):
        path['T'] = np.dot(path['T'], expm(u[i]))
        path['x'].append(path['T'][0, 2])
        path['y'].append(path['T'][1, 2])

    # build a 2D robot!
    robot = {}
    robot['dt'] = dt
    robot['x'] = np.eye(3)  # state mean
    robot['Cov'] = np.zeros([3, 3])  # covariance
    # motion model noise covariance
    robot['Q'] = np.diag([0.01 ** 2, 0.01 ** 2, 0.01 ** 2])
    # Cholesky factor of covariance for sampling
    robot['L'] = np.linalg.cholesky(robot['Q'])
    # se(2) generators; twist = vect(v1, v2, omega)
    robot['G1'] = np.array([[0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])
    robot['G2'] = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
    robot['G3'] = np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 0]])
    # SE(2) Adjoint
    robot['Ad'] = adjoint

    # create confidence ellipse
    # first create points from a unit circle + angle (third dimension of so(3))
    phi = np.arange(-np.pi, np.pi+0.01, 0.01)
    circle = np.array([np.cos(phi).reshape(-1, 1), np.sin(phi).reshape(-1, 1), np.zeros([len(phi), 1])]).reshape(3, -1)
    # Chi-squared 3-DOF 95% confidence (0.05):7.815
    scale = np.sqrt(7.815)

    # incremental visualization
    green = np.array([0.2980, 0.6, 0])
    crimson = np.array([220, 20, 60]) / 255
    darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    VermillionRed = np.array([156, 31, 46]) / 255
    DupontGray = np.array([144, 131, 118]) / 255

    fig = plt.figure()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-2, 12])
    plt.ylim([-2, 12])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    # plot Normal path
    plt.plot(path['x'], path['y'], '-', color=Darkgrey, linewidth=3, label='Normal path')

    # plot initial mean
    plt.plot(robot['x'][0, 2], robot['x'][1, 2], '.', color=green, alpha=0.5, markersize=5, label='Mean')

    # main loop; iterate over the control inputs and move the robot
    ELLIPSE = np.zeros([circle.shape[1], 2])  # covariance ellipse on manifold (nonlinear)
    ellipse = np.zeros([circle.shape[1], 2])  # covariance ellipse on Lie algebra
    plot_legend = True
    for i in range(len(u)):
        # move the robot based on the input
        robot = propagation(robot, u[i])
        robot['L'] = np.linalg.cholesky(robot['Cov'])
        for j in range(circle.shape[1]):
            # sample covariance on SE(2)
            ell_se2_vec = scale * np.dot(robot['L'], circle[:, j])
            # retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
            temp = np.dot(robot['x'], expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
                                           + robot['G3'] * ell_se2_vec[2]))
            ELLIPSE[j, :] = np.array([temp[0, 2], temp[1, 2]])
        if plot_legend:
            plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2, label='Covariance-SE(2)')
            plt.plot(robot['x'][0, 2], robot['x'][1, 2], 'o', color=crimson, alpha=0.7, markersize=8, label='Location')
            plot_legend = False
        else:
            plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2)
            plt.plot(robot['x'][0, 2], robot['x'][1, 2], 'o', color=crimson, alpha=0.7, markersize=8)
        plt.legend(loc='best')
        plt.pause(0.05)
    fig.savefig('banana_is_gaussian_rect.png')
    plt.show()


