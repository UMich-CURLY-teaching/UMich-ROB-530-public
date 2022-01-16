#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/31/2020
#

import numpy as np
from unscented_kalman_filter import unscented_kalman_filter
from extended_kalman_filter import extended_kalman_filter
import math
import matplotlib.pyplot as plt


# This function is used to wrap angles in radians to the interval [-pi, pi]
# pi maps to pi and -pi maps to -pi
def wrapToPi(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    while abs(x_wrap) > np.pi:
        x_wrap -= 2 * np.pi * np.sign(x_wrap)
    return x_wrap


class myStruct():
    pass


# process model
def process_model(x):
    f = np.array([[x[0]], [x[1]]])
    return f.reshape([2, 1])


# measurement model
def measurement_model(x):
    h = np.array([[np.sqrt(np.sum(x[0] ** 2 + x[1] ** 2))],
                  [math.atan2(x[0], x[1])]])
    return h.reshape([2, 1])


# measurement model Jacobian
def measurement_Jacobian(x):
    H = np.array([[x[0] / np.sqrt(np.sum(x[0] ** 2 + x[1] ** 2)), x[1] / np.sqrt(np.sum(x[0] ** 2 + x[1] ** 2))],
                  [x[1] / np.sum(x[0] ** 2 + x[1] ** 2), -x[0] / np.sum(x[0] ** 2 + x[1] ** 2)]])
    return H.reshape([2, 2])


# Toy example for tracking a single target using an Unscented Kalman filter and
# range-bearing measurements
if __name__ == '__main__':
    # First simulate a target that moves on a curved path; we assume ownship is at the origin (0, 0) and received noisy
    # range and bearing measurements of the target location. There is no knowledge of the target motion, hence, we
    # assume a random walk motion model

    # ground truth data
    gt = myStruct()
    gt.x = np.arange(-5, 5.1, 0.1)
    gt.y = np.array(1 * np.sin(gt.x) + 3)

    # measurements
    R = np.diag(np.power([0.05, 0.01], 2))
    # Cholesky factor of covariance for sampling
    L = np.linalg.cholesky(R)
    z = np.zeros([2, len(gt.x)])
    for i in range(len(gt.x)):
        # sample from a zero mean Gaussian with covariance
        # noise = np.dot(L, np.random.randn(2, 1)).reshape(-1)
        noise = [0.01, 0.02]
        z[:, i] = np.array([np.sqrt(gt.x[i] ** 2 + gt.y[i] ** 2), math.atan2(gt.x[i], gt.y[i])]) + noise

    # First we run EKF and then UKF for comparison
    # build the system
    sys = myStruct()
    sys.A = np.eye(2)
    sys.B = []
    sys.f = process_model
    sys.h = measurement_model
    sys.H = measurement_Jacobian
    sys.Q = 1e-3 * np.eye(2)
    sys.R = np.diag(np.power([0.05, 0.01], 2))

    # initialize the state using the first measurement
    init = myStruct()
    init.x = np.zeros([2, 1])
    init.x[0, 0] = z[0, 0] * np.sin(z[1, 0])
    init.x[1, 0] = z[0, 0] * np.cos(z[1, 0])
    init.Sigma = 1 * np.eye(2)

    ###################################################################
    # Extended Kalman Filter
    ekf = extended_kalman_filter(sys, init)
    x_ekf = []
    x_ekf.append(init.x)  # state
    # main loop; iterate over the measurement
    for i in range(1, np.shape(z)[1], 1):
        ekf.prediction()
        ekf.correction(z[:, i].reshape([2, 1]))
        x_ekf.append(ekf.x)

    x_ekf = np.array(x_ekf).reshape([-1, 2])

    ###################################################################
    # Unscented Kalman Filter
    init.kappa_f = 2
    init.kappa_h = 2
    ukf = unscented_kalman_filter(sys, init)
    x_ukf = []
    x_ukf.append(init.x)  # state
    # main loop; iterate over the measurements
    for i in range(1, np.shape(z)[1], 1):
        ukf.prediction()
        ukf.correction(z[:, i].reshape([2, 1]))
        x_ukf.append(ukf.x)
    x_ukf = np.array(x_ukf).reshape([-1, 2])

    # plotting
    fig = plt.figure()
    plt.grid(True)
    plt.axis('equal')
    line1, = plt.plot(0, 0, '^', color='blue', markersize=20)
    line2, = plt.plot(gt.x, gt.y, '-', linewidth=2)
    line3, = plt.plot(x_ekf[:, 0], x_ekf[:, 1], '-.r', linewidth=2)
    line4, = plt.plot(x_ukf[:, 0], x_ukf[:, 1], '--k', linewidth=1.5)
    plt.legend([line1, line2, line3, line4], [r'ownship', r'ground truth', r'EKF', r'UKF'], loc='best')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()








