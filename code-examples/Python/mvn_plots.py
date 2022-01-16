#!/usr/bin/env python

#
# Author: Fangtong Liu
# Date: 03/22/2020
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm


# MVN
mu = np.array([0, 0.5])  # mean
Sigma = np.array([[.8, .3], [.3, 1]])  # covariance
# compute PDF
x1 = np.arange(-3, 3.1, .1)
x2 = np.arange(-3, 4.1, .1)
X1, X2 = np.meshgrid(x1, x2)
X1X2 = np.transpose(np.array([X1.reshape(-1), X2.reshape(-1)]))
F = multivariate_normal.pdf(X1X2, mu, Sigma)
F = F.reshape((np.size(x2), np.size(x1)))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, F, cmap='viridis', linewidth=0, antialiased=False)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 4)
ax.set_zlim(0, .2)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'PDF')
# fig.colorbar(surf, shrink=0.25, aspect=6)
fig.savefig("mvn_surfave.png")
plt.show()

# marginals
f1 = norm.pdf(x1, mu[0], Sigma[0][0])
f2 = norm.pdf(x2, mu[1], Sigma[1][1])
# x_1 conditioned on x_2
mu_12 = mu[0] + Sigma[0][1] * (1 / Sigma[1][1]) * (0.9 - 0.5)  # conditional mean
sigma_12 = Sigma[0][0] - Sigma[0][1] * (1 / Sigma[1][1]) * Sigma[1][0]  # conditional variance
f12 = norm.pdf(x1, mu_12, sigma_12)
# marginal plot
fig = plt.figure()
line1, = plt.plot(x1, f1)
line2, = plt.plot(x2, f2, linestyle='--')
line3, = plt.plot(x1, f12, linestyle='-.')
plt.grid(True)
plt.xlabel(r'$x$')
plt.ylabel(r'PDF')
plt.legend([line1, line2, line3], [r'$p(x_1)$', r'$p(x_2)$', r'$p(x_1 | x_2 = 0.9)$'], loc='upper right')
plt.xlim(-4, 4)
plt.ylim(0, 0.6)
plt.text(-3.95, .57, r'$p(x_1) = \mathcal{N}(0, 0.8)$', size=12)
plt.text(-3.95, .54, r'$p(x_2) = \mathcal{N}(0.5, 1)$', size=12)
plt.text(-3.95, .51, r'$p(x_1 | x_2 = 0.9) = \mathcal{N}(0.12, 0.71)$', size=12)
fig.savefig("mvn_marginals_cond.pdf")
plt.show()

fig, ax = plt.subplots()
cp = ax.contourf(x1, x2, F, corner_mask="True", levels=30)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
fig.savefig("mvn_top.png")
plt.show()

fig, ax = plt.subplots()
cp = ax.contour(x1, x2, F, levels=10)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 4)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.clabel(cp, inline=True, fontsize=8, colors='black')
plt.text(-2.5, 3.25, r'$\mathcal{N}([0; 0.5], [0.8, 0.3; 0.3, 1])$')
fig.savefig("mvn_contour.pdf")
plt.show()

