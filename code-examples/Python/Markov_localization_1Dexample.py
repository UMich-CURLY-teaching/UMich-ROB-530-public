#!/usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/22/2020
#


import numpy as np
import matplotlib.pyplot as plt


# Motion model
# Motion model changes the belief based on action u
def motion_model(xi, xj, u):
    if u == 1:  # move forward
        dx = xi - xj
        if dx == 1 or dx == -19:
            p = 1
        else:
            p = 0
    elif u == 0:  # stay
        dx = xi - xj
        if dx == 0:
            p = 1
        else:
            p = 0
    # print(p)
    # else:
    #     assert (u == 1 or u == 0), 'The action is not defined'
    return p


# Measurement model
# Measurement model returns p(z|x) based on a likelihood map
def measurement_model(x, likelihood_map):
    return likelihood_map[:, x]


if __name__ == "__main__":
    #
    # Markov localization example in a 1D world
    # colors
    green = np.array([0.2980, 0.6, 0])
    darkblue = np.array([0, 0.2, 0.4])
    VermillionRed = np.array([156, 31, 46]) / 255

    #
    # Action set; 0: stay, 1: move forward
    # control inputs (sequence of actions)
    u = [1, 1, 1, 1, 1, 1, 1, 0]

    # Measurement
    z = [1, 0, 0, 0, 0, 1, 0, 0]

    # State space: the world has 20 cells, after 20 the robot will be at cell 1 again
    X = np.arange(0, 20, 1)

    # Belief initialization
    bel = np.ones([1, 20]) * 1 / 20  # uniform prior
    prior_bel = np.copy(bel)  # use for plot prior belief
    #
    # Likelihood map to provide measurements
    # The robot receives measurements at cell 4, 9, and 13
    likelihood_map = np.ones([1, 20]) * 0.2
    for i in [3, 8, 12]:
        likelihood_map[:, i] = 0.8

    #
    # Markov Localization using Bayes filter
    # This main loop can be run forever, but we run it for a limited sequence of control inputs
    k = 0  # step counter
    bel_predicted = np.ones([1, 20]) * 1 / 20  # predicted belief
    fig = plt.figure()
    while len(u):
        if z[k] == 1:  # measurement received
            eta = 0  # normalization constant
            for i in range(len(X)):
                likelihood = measurement_model(X[i], likelihood_map)  # get measurement likelihood
                bel[:, i] = likelihood * bel_predicted[:, i]  # unnormalized Bayes update
                eta = eta + bel[:, i]
            bel = bel / eta  # normalize belief

        # prediction; belief convolution
        for m in range(len(X)):
            bel_predicted[:, m] = 0
            for n in range(len(X)):
                pu = motion_model(X[m], X[n], u[0])
                bel_predicted[:, m] = bel_predicted[:, m] + pu * bel[:, n]

        # set the predicted belief as prior
        bel = np.copy(bel_predicted)

        # remove the executed action from the list
        u.remove(u[0])
        k = k + 1

        # plot prior belief
        plt.clf()
        plt.grid(True)
        width = 0.35  # set bar width
        ax1 = plt.subplot(311)
        ax1.set_aspect(3)
        plt.bar(X+1, prior_bel.reshape(-1), width, color=darkblue)
        plt.title(r'Prior Belief Map')
        plt.xticks(X+1)
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xlim(0.5, 20.5)
        plt.ylim(0, 1)
        plt.ylabel(r'$p(x)$')

        # plot likelihood
        ax2 = plt.subplot(312)
        ax2.set_aspect(3)
        plt.bar(X+1, likelihood_map.reshape(-1), width, color=green)
        plt.title(r'Likelihood Map')
        plt.xticks(X+1)
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xlim(0.5, 20.5)
        plt.ylim(0, 1)
        plt.ylabel(r'$p(z|x)$')

        # plot posterior belief
        ax3 = plt.subplot(313)
        ax3.set_aspect(3)
        plt.bar(X+1, bel.reshape(-1), width, color=VermillionRed)
        plt.title(r'Posterior Belief Map')
        plt.xlim(0.5, 20.5)
        plt.ylim(0, 1)
        plt.ylabel(r'$p(x|z)$')
        plt.xticks(X+1)
        plt.yticks(np.arange(0, 1.2, 0.2))
        if len(u) == 0:
            plt.show()
        else:
            plt.pause(0.5)


