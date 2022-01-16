#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/31/2020
#

import numpy as np


# Unscented transform class for uncertainty propagation through a nonlinear model.
# For the algorithm, see State Estimation for Robotics, Timothy D. Barfoot, 2018, Chapter 4
class unscented_transform:
    def __init__(self, *args):
        # unscented_transform construct at instance of this class
        #
        # Input:
        #   mean, cov, f, kappa
        if len(args) == 0:
            self.x_input = []  # input mean
            self.P = []  # input covariance
            self.func = []  # nonlinear model
            self.n = []  # input dimension
            self.kappa = []  # user-defined parameter to control the sigma points
        elif len(args) == 4:
            self.x_input = args[0]
            self.P = args[1]
            self.func = args[2]
            self.n = len(args[0])
            self.kappa = args[3]
        else:
            raise SystemExit('Input must be mean, cov, function, and kappa')

    def set(self, mean, cov, f, kappa):
        # set the input after the instance constructed and used
        self.x_input = mean
        self.P = cov
        self.func = f
        self.n = len(mean)
        self.kappa = kappa

    def sigma_points(self):
        # sigma points around the reference point
        self.L = np.sqrt(self.n + self.kappa) * np.linalg.cholesky(self.P)  # scaled Cholesky factor of P
        Y = self.x_input.repeat(len(self.x_input), axis=1)
        self.X = np.hstack((self.x_input, Y + self.L, Y - self.L))  # 2n+1 sigma points
        self.w = np.zeros([2 * self.n + 1, 1])  # 2n+1 sigma points weights
        self.w[0] = self.kappa / (self.n + self.kappa)
        self.w[1:] = 1 / (2 * (self.n + self.kappa))
        self.w = self.w.reshape(-1)


    def propagate(self):
        # propagate the input Gaussian using an unscented transform
        self.sigma_points()
        # compute sample mean and covariance
        self.mean = 0
        self.Cov = 0
        self.Y = []
        for i in range(2*self.n + 1):
            Yvalue = self.func(self.X[:, i])
            self.Y.append(Yvalue)
            self.mean = self.mean + self.w[i] * Yvalue

        self.Y = (np.array(self.Y).reshape([2*self.n+1, -1])).T
        temp = self.Y - self.mean.reshape(2, 1)
        self.Cov = np.dot(np.dot(temp, np.diag(self.w)), temp.T)
        self.Cov_xy = np.dot(np.dot(self.X - self.x_input, np.diag(self.w)), (self.Y - self.mean.reshape(2, 1)).T)


