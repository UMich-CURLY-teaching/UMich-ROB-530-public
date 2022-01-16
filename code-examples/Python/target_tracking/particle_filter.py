#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 04/02/2020
#

import numpy as np
from numpy.random import randn, rand
from scipy.stats import multivariate_normal


class myStruct():
    def __init__(self):
        self.x = []
        self.w = []


# This function is used to wrap angles in radians to the interval [-pi, pi]
# pi maps to pi and -pi maps to -pi
def wrapToPI(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    while abs(x_wrap) > np.pi:
        x_wrap -= 2 * np.pi * np.sign(x_wrap)
    return x_wrap

# Particle filter class for state estimation of a nonlinear system
# The implementation follows the Sample Importance Resampling (SIR)
# filter a.k.a bootstrap filter
class particle_filter:

    def __init__(self, system, init):
        # Particle filter construct an instance of this class
        #
        # Input:
        #   system: system and noise models
        #   init:   initialization parameters

        self.f = system.f  # process model
        self.Q = system.Q  # measurement model
        self.LQ = np.linalg.cholesky(self.Q)  # Cholesky factor of Q
        self.h = system.h  # measurement model
        self.R = system.R  # measurement noise covariance
        self.n = init.n  # number of particles

        # initialize particles
        self.p = myStruct()  # particles

        wu = 1 / self.n  # uniform weights
        L_init = np.linalg.cholesky(init.Sigma)
        for i in range(self.n):
            self.p.x.append(np.dot(L_init, randn(len(init.x), 1)) + init.x)
            # self.p.x.append(np.dot(L_init, 0.5 * np.ones([len(init.x), 1])) + init.x)
            self.p.w.append(wu)
        self.p.x = np.array(self.p.x).reshape(-1, len(init.x))
        self.p.w = np.array(self.p.w).reshape(-1, 1)

    def sample_motion(self):
        # A simple random walk motion model
        for i in range(self.n):
            # sample noise
            w = np.dot(self.LQ, randn(2, 1))
            # w = np.dot(self.LQ, np.array([[0.5], [0.01]]))
            # propagate the particle
            self.p.x[i, :] = self.f(self.p.x[i, :], w).reshape(-1)

    def sample_motion_cv(self):
        # A constant velocity random walk motion model
        for i in range(self.n):
            # sample noise
            w = np.dot(self.LQ, randn(4, 1))
            # propagate the particle
            self.p.x[i, :] = self.f(self.p.x[i, :], w).reshape(-1)

    def importance_measurement(self, z):
        # compare important weight for each particle based on the obtain range and bearing measurements
        #
        # Inputs:
        #   z: measurement
        w = np.zeros([self.n, 1])  # importance weights
        for i in range(self.n):
            # compute innovation statistics
            # We know here z[1] is an angle
            v = z - self.h(self.p.x[i, :])
            v[1] = wrapToPI(v[1])
            w[i] = multivariate_normal.pdf(v.reshape(-1), np.array([0, 0]), self.R)

        # update and normalize weights
        self.p.w = np.multiply(self.p.w, w)  # since we used motion model to sample
        self.p.w = self.p.w / np.sum(self.p.w)
        # compute effective number of particles
        self.Neff = 1 / np.sum(np.power(self.p.w, 2))  # effective number of particles

    def resampling(self):
        # low variance resampling
        W = np.cumsum(self.p.w)
        r = rand(1) / self.n
        # r = 0.5 / self.n
        j = 1
        for i in range(self.n):
            u = r + (i - 1) / self.n
            while u > W[j]:
                j = j + 1
            self.p.x[i, :] = self.p.x[j, :]
            self.p.w[i] = 1 / self.n






