#!/usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/31/2020
#

import numpy as np


# This function is used to wrap angles in radians to the interval [-pi, pi]
# pi maps to pi and -pi maps to -pi
def warpToPi(phase):
    x_wrap = np.remainder(phase, 2 * np.pi)
    while np.abs(x_wrap) > np.pi:
        x_wrap -= 2 * np.pi * np.sign(x_wrap)
    return x_wrap


# Extended Kalman filter class for state estimation of a nonlinear system
class extended_kalman_filter:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        #
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.A = system.A  # system matrix Jacobian
        self.B = system.B  # input matrix Jacobian
        self.f = system.f  # process model
        self.H = system.H  # measurement model Jacobian
        self.Q = system.Q  # input noise covariance
        self.R = system.R  # measurement noise covariance
        self.h = system.h  # measurement model
        self.x = init.x  # state vector
        self.Sigma = init.Sigma  # state covariance

    def prediction(self):
        # EKF propagation (prediction) step
        self.x_pred = self.f(self.x)  # predicted state
        self.Sigma_pred = np.dot(np.dot(self.A, self.Sigma), self.A.T) + self.Q  # predicted state covariance
        self.z_hat = self.h(self.x_pred)  # predicted measurement

    def correction(self, z):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement

        # evaluate measurement Jacobian at current operating point
        H = self.H(self.x_pred)

        # compute innovation statisticsf
        # We know here z[1] is an angle
        self.v = z - self.z_hat  # innovation
        self.v[1] = warpToPi(self.v[1])
        self.S = np.dot(np.dot(H, self.Sigma_pred), H.T) + self.R  # innovation covariance

        # filter gain
        self.K = np.dot(np.dot(self.Sigma_pred, H.T), np.linalg.inv(self.S))  # Kalman (filter) gain

        # correct the predicted state statistics
        self.x = self.x_pred + np.dot(self.K, self.v)
        I = np.eye(np.shape(self.x)[0])
        temp = I - np.dot(self.K, H)
        self.Sigma = np.dot(np.dot(temp, self.Sigma_pred), temp.T) + np.dot(np.dot(self.K, self.R), self.K.T)  # Joseph update form