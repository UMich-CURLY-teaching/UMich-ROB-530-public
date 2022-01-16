#!/usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/31/2020
#

import numpy as np
import math
import matplotlib.pyplot as plt
from extended_kalman_filter import extended_kalman_filter


class myStruct:
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
    H = np.array([[x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2), x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2)],
                  [x[1] / (x[0] ** 2 + x[1] ** 2), -x[0] / (x[0] ** 2 + x[1] ** 2)]])
    return H.reshape([2, 2])



#
# Toy example for tracking a single target using an EKF and range-bearing measurements
#
if __name__ == "__main__":
    # First simulate a target that moves on a curved path; we assume ownship at the origin(0, 0)
    # and received noisy range and bearing measurements of the target location. There is no knowledge
    # of the target motion, hence, we assume a random walk model

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
        noise = np.dot(L, np.random.randn(2, 1)).reshape(-1)
        z[:, i] = np.array([np.sqrt(gt.x[i] ** 2 + gt.y[i] ** 2), math.atan2(gt.x[i], gt.y[i])]) + noise

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

    ekf = extended_kalman_filter(sys, init)
    x = []
    x.append(init.x)  # state
    # main loop; iterate over the measurement
    for i in range(1, np.shape(z)[1], 1):
        ekf.prediction()
        ekf.correction(z[:, i].reshape([2, 1]))
        x.append(ekf.x)

    x = np.array(x)

    # plotting
    fig = plt.figure()
    line1, = plt.plot(0, 0, '^', color='b', markersize=20)
    line2, = plt.plot(gt.x, gt.y, '-', linewidth=2)
    line3, = plt.plot(x[:, 0, :], x[:, 1, :], '-m', linewidth=1.5)
    plt.legend([line1, line2, line3], [r'ownship', r'ground truth', r'EKF'], loc='best')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid(True)
    plt.show()

        




