#! usr/bin/env python

#
# occupancy grid mapping example
# Author: Fangtong Liu
# Date: 04/18/2020
#

import numpy as np
from scipy.io import loadmat
from occupancy_grid_map import occupanvy_grid_map
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataLoad = loadmat('sample_Intel_dataset.mat')
    robotPose = dataLoad['robotPose']
    laserScan = dataLoad['laserScan']
    globalLaserScan = dataLoad['globalLaserScan']

    # color
    green = np.array([0.2980, 0.6, 0])
    crimson = np.array([220, 20, 60]) / 255
    darkgrey = [0.35, 0.35, 0.35]

    ogm = occupanvy_grid_map()
    ogm.occupancy_grid_map_construct(robotPose, laserScan)
    ogm.build_ogm()
    ogm.plot()

    print('Plotting dataset')
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(robotPose['x'][0][0], robotPose['y'][0][0], linewidth=2)
    for i in range(len(laserScan)):
        ax.plot(globalLaserScan[i][0][0, :], globalLaserScan[i][0][1, :], '.', color=darkgrey, alpha=0.9, markersize=4)
    ax.plot(robotPose['x'][0][0][0], robotPose['y'][0][0][0], 's', color=green, alpha=0.8, markersize=14)
    ax.plot(robotPose['x'][0][0][-1], robotPose['y'][0][0][-1], 'o', color=crimson, alpha=0.8, markersize=14)
    plt.axis('equal')
    plt.tight_layout()
    fig.savefig('ptcloud_intel.png')
    plt.show()







