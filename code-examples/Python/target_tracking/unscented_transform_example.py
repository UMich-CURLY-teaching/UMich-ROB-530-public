#!usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/31/2020
#

import numpy as np
import matplotlib.pyplot as plt
from unscented_transform import unscented_transform


# build a nonlinear map
def f(x):
    output = np.array([[x[0] * np.cos(x[1])], [x[0] * np.sin(x[1])]])
    return output.reshape([2, -1])


# An example of using the unscented transform for propagating a Gaussian distribution through a nonlinear map.
# We use polar to Cartesian for this code.
if __name__ == "__main__":

    # create a random mean and covariance
    x = np.array([[1.5], [np.pi / 6]])  # x = (r, theta)
    # P = ap.array([[0.3 ** 2, -0.14 ** 2], [-0.14 ** 2, 0.35 ** 2]])
    P = np.array([[0.1 ** 2, -0.09 ** 2], [-0.09 ** 2, 0.6 ** 2]])

    # propagate the uncertainty using UT and affine model to compare
    kappa = 2
    ut = unscented_transform(x, P, f, kappa)
    ut.propagate()

    # visualization

    # colors
    green = np.array([0.298, 0.6, 0])
    # crimson = np.array([220, 20, 60]) / 255
    # darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    # darkgrey = np.array([0.35, 0.35, 0.35])
    # lightgrey = np.array([0.7, 0.7, 0.7])
    # Lightgrey = np.array([0.9, 0.9, 0.9])
    VermillionRed = np.array([156, 31, 46]) / 255
    DupontGray = np.array([144, 131, 118]) / 255
    # Azure = np.array([53, 112, 188]) / 255
    # purple = np.array([178, 102, 255]) / 255
    # orange = np.array([255, 110, 0]) / 255

    # create confidence ellipse
    # first create points from unit circle
    phi = np.arange(-np.pi, np.pi+0.1, 0.1)
    circle = np.array([np.cos(phi), np.sin(phi)])
    # Chi-square 2-DOF 95% confidence (0.05):5.991
    scale = np.sqrt(5.991)
    # apply the transformation and scale of the covariance
    ellipse_polar = (np.dot(scale * np.linalg.cholesky(P), circle) + x).T
    ellipse_cartesian = (np.dot(scale * np.linalg.cholesky(ut.Cov), circle) + ut.mean).T

    # generate samples for bmultivariate_normaloth polar and cartesian coordinates
    s_polar = (np.dot(np.linalg.cholesky(P), np.random.randn(2, 1000)) + x).T
    s_cartesian = np.zeros(np.shape(s_polar))
    for i in range(s_polar.shape[0]):
        s_cartesian[i, :] = np.reshape(f(s_polar[i, :]), -1)

    # plot in polar coordinates
    fig1 = plt.figure()
    plt.grid(True)
    plt.axis('equal')
    line1, = plt.plot(s_polar[:, 0], s_polar[:, 1], '.', color=DupontGray, markersize=4)
    line2, = plt.plot(x[0], x[1], 'o', color=VermillionRed, markersize=18, fillstyle='none')
    line3, = plt.plot(ellipse_polar[:, 0], ellipse_polar[:, 1], color=VermillionRed, linewidth=2)
    line4, = plt.plot(ut.X[0, :], ut.X[1, :], '.', color=Darkgrey, markersize=18)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\theta$')
    plt.legend([line1, line2, line3, line4], [r'Samples', r'Mean', r'$95\%$ Confidence Ellipse', r'Sigma Points'],
               loc='lower right', bbox_to_anchor=(1.1, 0))
    plt.text(1.75, 1.6, r'$\kappa = 2$', fontsize=18)
    plt.show()

    # plot in Cartesian coordinates
    fig2 = plt.figure()
    plt.grid(True)
    plt.axis('equal')
    line1, = plt.plot(s_cartesian[:, 0], s_cartesian[:, 1], '.', color=DupontGray, markersize=4)
    line2, = plt.plot(ut.mean[0], ut.mean[1], 'o', color=VermillionRed, markersize=18, fillstyle='none')
    line3, = plt.plot(ellipse_cartesian[:, 0], ellipse_cartesian[:, 1], color=VermillionRed, linewidth=2)
    line4, = plt.plot(ut.Y[0, :], ut.Y[1, :], '.', color=Darkgrey, markersize=18)
    plt.xlabel(r'$x=r\cos(\theta)$')
    plt.ylabel(r'$y=r\sin(\theta)$')
    plt.legend([line1, line2, line3, line4], [r'Samples', r'Mean', r'$95\%$ Confidence Ellipse', r'Sigma Points'],
               loc='lower right', bbox_to_anchor=(1.1, 0))
    plt.text(1.6, 1.8, r'$\kappa=2$', fontsize=18)
    plt.show()





