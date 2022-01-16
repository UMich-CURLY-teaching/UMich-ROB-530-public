#!usr/bin/env python

#
# Right-Invariant Extended Kalman filter class for 2D Localization, SE(2).
#
# Author: Fangtong Liu
# Date: 05/25/2020
#

import numpy as np
from scipy.linalg import block_diag, expm


class Right_IEKF:

    def __init__(self, system):
        # Right_IEKF Construct an instance of this class
        #
        # Input:
        #   system:     system and noise models
        self.A = system['A']  # error dynamics matrix
        self.f = system['f']  # process model
        self.H = system['H']  # measurement error matrix
        self.Q = system['Q']  # input noise covariance
        self.N = system['N']  # measurement noise covariance
        self.X = np.eye(3)  # state vector
        self.P = 0.1 * np.eye(3)  # state covariance

    def Ad(self, X):
        # Adjoint
        AdX = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
        AdX = np.vstack((AdX, np.array([0, 0, 1])))
        return AdX

    def wedge(self, x):
        # wedge operation for se(2) to put an R^3 vector to the Lie algebra basis
        G1 = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])  # omega
        G2 = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0]])  # v_1
        G3 = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 0, 0]])  # v_2
        xhat = G1 * x[0] + G2 * x[1] + G3 * x[2]
        return xhat

    def prediction(self, u):
        # EKF propagation (prediction) step
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + np.dot(np.dot(self.Ad(self.X), self.Q), self.Ad(self.X).T)
        self.X = self.f(self.X, u)

    def correction(self, Y1, b1, Y2, b2):
        # RI-EKF correction Step
        H = np.vstack((self.H(b1-1), self.H(b2-1)))  # stack H
        h1 = H[0:2, :]
        h2 = H[3:5, :]
        H = np.vstack((h1, h2))  # 4 x 3 matrix, remove zero rows
        N = np.dot(np.dot(self.X, block_diag(self.N, 0)), self.X.T)
        N = block_diag(N[0:2, 0:2], N[0:2, 0:2])  # 4 x 4 block-diagonal matrix
        # filter gain
        S = np.dot(np.dot(H, self.P), H.T) + N
        L = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update state
        nu = np.dot(block_diag(self.X, self.X), np.vstack((Y1, Y2))) - np.vstack((b1.reshape(-1, 1), b2.reshape(-1, 1)))
        nu = np.hstack((nu[0:2, 0], nu[3:5, 0]))
        delta = self.wedge(np.dot(L, nu))  # innovation in the spatial frame
        self.X = np.dot(expm(delta), self.X)

        # Update Covariance
        I = np.eye(np.shape(self.P)[0])
        temp = I - np.dot(L, H)
        self.P = np.dot(np.dot(temp, self.P), temp.T) + np.dot(np.dot(L, N), L.T)





