#!/usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/22/2020
#

import numpy as np
import random
import math
import matplotlib.pyplot as plt


# nice colors
green = np.array([0.2980, .6, 0])
crimson = np.array([220, 20, 60]) / 255
darkblue = np.array([0, .2, .4])
Darkgrey = np.array([.25, .25, .25])
darkgrey = np.array([.35, .35, .35])
lightgrey = np.array([.7, .7, .7])
Lightgrey = np.array([.9, .9, .9])
VermillionRed = np.array([156, 31, 46]) / 255
DupontGray = np.array([144, 131, 118]) / 255
Azure = np.array([53, 112, 188]) / 255
purple = np.array([178, 102, 255]) / 255
orange = np.array([255, 110, 0]) / 255

##################### 2D Example ##########################
# MVN
mu = np.array([0, 0.5])  # mean
Sigma = np.array([[0.8, -0.3], [-0.3, 1]])  # covariance

# Draw samples
d = 2  # dimension of data
n = 1000  # number of samples
X = np.zeros([d, n])
L = np.linalg.cholesky(Sigma)  # Cholesky factor of covariance
for i in range(n):
    Z = [random.normalvariate(0, 1) for j in range(d)]  # draw a d-dimensional vector of standard normal random variables
    X[:, i] = np.dot(L, Z) + mu
X = X.T  # make rows to be samples/observations

# Sample mean
mu_bar = (1 / n * np.sum(X, axis=0)).T  # transpose to make it a column vector

print('\n2D Example')
print('mean', '\t\t\t\tsample mean')
print(mu, '\t\t', mu_bar)

# Sample covariance
e = (X - mu_bar).T  # centralize samples around sample mean
Sigma_bar = np.dot(e, e.T) / (n-1)
# Alternative option using numpy.cov
# Sigma_bar = np.cov(X.T)

print('covariance', '\t\t\tsample covariance')
for i in range(np.shape(Sigma)[1]):
    print(Sigma[i, :], '\t', Sigma_bar[i, :])

# create confidence ellipse
# first create points from a unit circle
phi = np.arange(-math.pi, math.pi+0.01, 0.01)
circle = np.array([np.cos(phi), np.sin(phi)])

# Chi-squared 2-DOF 95% percent confidence (0.05):5.991
scale = math.sqrt(5.991)
# apply the transformation and scale of the covariance
ellipse = np.dot(np.dot(scale, L), circle).T + mu
# test plot to visualize circle and ellipse together
# fig = plt.figure()
# plt.axis('equal')
# plt.grid(True)
# plt.plot(circle[0, :], circle[1, :])
# plt.plot(ellipse[:, 0], ellipse[:, 1])
# plt.plot(mu[0], mu[1], 's', markersize=12)
# plt.show()

fig = plt.figure()
h1, = plt.plot(X[:, 0], X[:, 1], '.', color=darkblue, markersize=4)
h2, = plt.plot(ellipse[:, 0], ellipse[:, 1], color=VermillionRed, linewidth=2)
h3, = plt.plot(mu_bar[0], mu_bar[1], 'sr', markerfacecolor='r', markersize=10)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend([h1, h2, h3], [r'Samples', r'$95\%$ Confidence Ellipse', 'Sample Mean'], loc='best')
plt.text(-0.5, 3.65, r'$\mathcal{N}([0; 0.5], [0.8, -0.3; -0.3, 1]$', fontsize=12)
plt.grid(True)
fig.savefig("confidence_ellipse.pdf")
plt.show()


##################### 3D Example ##########################
# MVN
mu = np.array([0, 0.5, 1])  # mean
Sigma = np.array([[0.8, -0.3, 0.1], [-0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])  # covariance

# Draw samples
d = 3  # dimension of data
n = 1000  # number of samples
X = np.zeros([d, n])
L = np.linalg.cholesky(Sigma)  # Cholesky factor of covariance
for i in range(n):
    Z = [random.normalvariate(0, 1) for j in range(d)]  # draw a d-dimensional vector of standard normal random variables
    X[:, i] = np.dot(L, Z) + mu
X = X.T  # make rows to be samples/observations

# Sample covariance
mu_bar = (1/n * np.sum(X, axis=0)).T  # transpose to make it a column vector

print('\n3D Example')
print('mean', '\t\t\t\tsample mean')
print(mu, '\t', mu_bar)

# Sample covariance
e = (X - mu_bar).T
Sigma_bar = np.dot(e, e.T) / (n-1)
# Alternative option using numpy.cov
# Sigma_bar = np.cov(X.T)

print('covariance', '\t\t\tsample covariance')
for i in range(d):
    print(Sigma[i, :], '\t', Sigma_bar[i, :])

# create confidence ellipsoid
# first create points from a unit sphere
phi = np.linspace(-math.pi, math.pi, 1000)
theta = np.linspace(-math.pi/2, math.pi/2, 1000)
PHI, THETA = np.meshgrid(phi, theta)
X_sph = np.multiply(np.cos(THETA), np.cos(PHI))
Y_sph = np.multiply(np.cos(THETA), np.sin(PHI))
Z_sph = np.sin(THETA)
sphere = np.array([X_sph.reshape(-1), Y_sph.reshape(-1), Z_sph.reshape(-1)])

# Chi-squared 3-DOF 95% percent confidence (0.05):7.815
scale = np.sqrt(7.815)
# apply the transformation and scale of the covariance
ellipsoid = np.dot(np.dot(scale, L), sphere).T + mu
# extract x, y, z matrices for plotting
X_ell = ellipsoid[:, 0].reshape(np.shape(X_sph))
Y_ell = ellipsoid[:, 1].reshape(np.shape(Y_sph))
Z_ell = ellipsoid[:, 2].reshape(np.shape(Z_sph))
# test plot to visualize sphere and ellipsoid together:
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# plt.grid(True)
# surf1 = ax.plot_surface(X_sph, Y_sph, Z_sph, color=green, alpha=0.4)
# surf2 = ax.plot_surface(X_ell, Y_ell, Z_ell, color=VermillionRed, alpha=0.4)
# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
point1 = ax.scatter(X[:, 0], X[:, 1], X[:, 2], '.', c='darkblue', s=2, alpha=0.3, label=r'Samples')
surf = ax.plot_surface(X_ell, Y_ell, Z_ell, color=VermillionRed, alpha=0.3, label=r'$95\%$ Confidence Ellipsoid')
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
point2 = ax.scatter(mu_bar[0], mu_bar[1], mu_bar[2], 's', c='r', s=30, label=r'Sample Mean')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
ax.legend()
ax.grid(True)
fig.savefig('confidence_ellipsoid_3d.png')
plt.show()

