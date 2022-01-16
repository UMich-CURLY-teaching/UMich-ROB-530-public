#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 04/02/2020
#

import numpy as np
from numpy.random import randn
from particle_filter import particle_filter
import matplotlib.pyplot as plt
import math


class myStruct:
    pass

# process model
def process_model(x, w):
    dt = 1
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    f = np.dot(F, x) + w.reshape(-1)
    return f.reshape([4, 1])

# measurement model
def measurement_model(x):
    h = np.array([[np.sqrt(x[0] ** 2 + x[1] ** 2)], [math.atan2(x[0], x[1])]])
    return h.reshape([2, 1])
# Toy example for tracking a single target using a particle filter and range-bearing measurements

if __name__ == '__main__':
    # First simulate a target that moves on a curved path; we assume ownship is at the origin (0, 0)
    # and received noisy range and bearing measurements of the target location. There is no knowledge
    # of the target motion, but we assume a constant velocity random walk model.

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
        # sample from a zero mean Gaussian with covariance R
        noise = np.dot(L, randn(2, 1)).reshape(-1)
        # noise = np.dot(L, np.array([[0.05], [0.1]])).reshape(-1)
        z[:, i] = np.array([np.sqrt(gt.x[i] ** 2 + gt.y[i] ** 2), math.atan2(gt.x[i], gt.y[i])]) + noise

    # build the system
    sys = myStruct()

    sys.f = process_model
    sys.h = measurement_model
    sys.Q = np.diag([1e-1, 1e-1, 1e-2, 1e-2])
    sys.R = np.diag(np.power([0.05, 0.01], 2))

    # initialization
    init = myStruct()
    init.n = 1000
    init.x = np.zeros([4, 1])
    init.x[0, 0] = z[0, 0] * np.sin(z[1, 0])
    init.x[1, 0] = z[0, 0] * np.cos(z[1, 0])
    init.Sigma = 1 * np.eye(4)

    filter = particle_filter(sys, init)
    x = np.empty([4, np.shape(z)[1]])  # state
    x[:, 0] = [np.nan, np.nan, np.nan, np.nan]

    # incremental visualization
    # nice colors
    green = np.array([0.2980, .6, 0])
    # crimson = np.array([220, 20, 60]) / 255
    # darkblue = np.array([0, .2, .4])
    # Darkgrey = np.array([.25, .25, .25])
    # darkgrey = np.array([.35, .35, .35])
    # lightgrey = np.array([.7, .7, .7])
    # Lightgrey = np.array([.9, .9, .9])
    # VermillionRed = np.array([156, 31, 46]) / 255
    # DupontGray = np.array([144, 131, 118]) / 255
    # Azure = np.array([53, 112, 188]) / 255
    # purple = np.array([178, 102, 255]) / 255
    # orange = np.array([255, 110, 0]) / 255

    # plotting
    fig1 = plt.figure()
    # ax = fig1.add_subplot(111)
    # ax.set_xlabel(r'$x_1$')
    # ax.set_ylabel(r'$x_2$')
    line1, = plt.plot(0, 0, '^', color='blue', markersize=20)
    line2, = plt.plot(gt.x, gt.y, '-', linewidth=2)
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-6, 6])
    plt.ylim([-1, 5])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    # plt.ion()
    hp, = plt.plot(filter.p.x[:, 0], filter.p.x[:, 1], '.', color=green, alpha=0.5)

    # main loop; iterate over the measurements
    for i in range(1, np.shape(z)[1], 1):
        filter.sample_motion_cv()
        filter.importance_measurement(z[:, i].reshape([2, 1]))
        if filter.Neff < filter.n / 5:
            filter.resampling()
        wtot = np.sum(filter.p.w)
        if wtot > 0:
            a = filter.p.x
            b = filter.p.w
            x[0, i] = np.sum(filter.p.x[:, 0] * filter.p.w.reshape(-1)) / wtot
            x[1, i] = np.sum(filter.p.x[:, 1] * filter.p.w.reshape(-1)) / wtot
            x[2, i] = np.sum(filter.p.x[:, 2] * filter.p.w.reshape(-1)) / wtot
            x[3, i] = np.sum(filter.p.x[:, 3] * filter.p.w.reshape(-1)) / wtot
        else:
            print('\033[91mWarning: Total weight is zero or nan!\033[0m')
            x[:, i] = [np.nan, np.nan, np.nan, np.nan]
        plt.clf()
        line1, = plt.plot(0, 0, '^', color='blue', markersize=20)
        line2, = plt.plot(gt.x, gt.y, '-', linewidth=2)
        hp, = plt.plot(filter.p.x[:, 0], filter.p.x[:, 1], '.', color=green, alpha=0.5, markersize=3)
        plt.grid(True)
        plt.axis('equal')
        plt.xlim([-6, 6])
        plt.ylim([-1, 5])
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.pause(0.05)

    # plotting
    fig2 = plt.figure()
    line4, = plt.plot(0, 0, '^', color='b', markersize=20)
    line5, = plt.plot(gt.x, gt.y, '-', linewidth=2)
    line6, = plt.plot(x[0, :], x[1, :], '-k', linewidth=1)
    plt.legend([line4, line5, line6], [r'ownship', r'ground truth', r'PF'], loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim([-6, 6])
    plt.ylim([-1, 5])
    plt.show()

    fig3 = plt.figure()
    line7, = plt.plot(x[2, :], linewidth=2)
    line8, = plt.plot(x[3, :], linewidth=2)
    plt.legend([line7, line8], [r'velocity-$x_1$', r'velocity-$x_2$'], loc='best')
    plt.xlabel('time')
    plt.ylabel('velocity')
    plt.grid(True)
    plt.gca().set_aspect('auto')
    plt.show()



