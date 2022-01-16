#!/usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/30/2020
#

import numpy as np


# Kalman filter class for state estimation of a linear system
class kalman_filter:

    def __init__(self, system, init):
        # KALMAN_FILTER Construct an instance of this class
        #
        # Inputs:
        #   system: linear system and noise models
        #   init:   initial state mean and covariance

        self.A = system.A  # system matrix
        self.B = system.B  # input matrix
        self.H = system.H  # measurement model
        self.Q = system.Q  # input noise covariance
        self.R = system.R  # measurement noise covariance
        self.x = init.x  # state vector
        self.Sigma = init.Sigma  # state covariance

    def prediction(self):
        # Kalman filter propagation (prediction) step

        self.x_pred = np.dot(self.A, self.x)  # + np.dot(self.B, u) # predicted state
        self.Sigma_pred = np.dot(np.dot(self.A, self.Sigma), np.transpose(self.A)) + self.Q  # predicted state covariance
        self.z_hat = np.dot(self.H, self.x_pred)  # predicted measurement
        # print("x_pred", self.x_pred)

    def correction(self, z):
        # Kalman filter correction step
        #
        # Inputs:
        #   z:  measurement

        # compute innovation statistics
        self.v = z - self.z_hat  # innovation
        self.S = np.dot(np.dot(self.H, self.Sigma_pred), np.transpose(self.H)) + self.R  # innovation covariance

        # filter gain
        self.K = np.dot(np.dot(self.Sigma_pred, np.transpose(self.H)), np.linalg.inv(self.S))  # Kalman (filter) gain

        # correct the predicted state statistics
        self.x = self.x_pred + np.dot(self.K, self.v)
        I = np.eye(np.shape(self.x)[0])
        temp = np.transpose(I - np.dot(self.K, self.H))
        self.Sigma = np.dot(np.dot(temp, self.Sigma_pred), temp.T) + np.dot(np.dot(self.K, self.R), np.transpose(self.K))  # Joseph update form