#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/31/2020
#

import numpy as np
from unscented_transform import unscented_transform


# This function is used to wrap angles in radians to the interval [-pi, pi]
# pi maps to pi and -pi maps to -pi
def wrapToPi(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    while abs(x_wrap) > np.pi:
        x_wrap -= 2 * np.pi * np.sign(x_wrap)
    return x_wrap


# Unscented transform class for uncertainty
# Unscented Kalman filter class for state estimation of a nonlinear system
class unscented_kalman_filter:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        self.f = system.f  # process model
        self.Q = system.Q  # input noise covariance
        self.R = system.R  # measurement noise covariance
        self.h = system.h  # measurement model
        self.x = init.x  # state vector
        self.Sigma = init.Sigma  # state covariance
        self.ut = unscented_transform()  # UT instance for prediction and correction
        self.kappa_f = init.kappa_f  # user-defined parameter to control the sigma points
        self.kappa_h = init.kappa_h  # user-defined parameter to control the sigma points

    def prediction(self):
        # UKF propagation (prediction) step
        self.ut.set(self.x, self.Sigma, self.f, self.kappa_f)
        self.ut.propagate()
        self.x_pred = self.ut.mean  # predicted state
        self.Sigma_pred = self.ut.Cov + self.Q  # predicted state covariance
        # compute new sigma points to predict the measurement
        self.ut.set(self.x_pred, self.Sigma_pred, self.h, self.kappa_h)
        self.ut.propagate()
        self.z_hat = self.ut.mean  # predicted measurement

    def correction(self, z):
        # UKF correction step
        #
        # Input:
        #   z:  measurement

        # compute innovation statistics
        # We know here z[1] is an angle
        self.v = z - self.z_hat  # innovation
        self.v[1] = wrapToPi(self.v[1])
        self.S = self.ut.Cov + self.R  # innovation covariance

        # compute state-measurement cross covariance
        self.Cov_xz = self.ut.Cov_xy  # state-measurement cross covariance

        # filter gain
        self.K = np.dot(self.Cov_xz, np.linalg.pinv(self.S))  # Kalman (filter) gain

        # correct the predicted state statistics
        self.x = self.x_pred + np.dot(self.K, self.v)
        self.Sigma = self.Sigma_pred - np.dot(np.dot(self.K, self.S), self.K.T)

