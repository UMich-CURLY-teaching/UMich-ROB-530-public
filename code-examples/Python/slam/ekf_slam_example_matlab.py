#!/usr/bin/env python

#
# EKF SLAM example using known correspondences and linear motion and measurement models
# AUthor: Fangtong Liu
# Date: 04/22/2020
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
# from sklearn.neighbors import KDTree
from scipy.linalg import block_diag


def positiveCheck(A):
    if np.all(np.linalg.eigvals(A) > 0):
        return True
    else:
        return False

def sort_idxlist(idx, map, pt):
    map_idx = map[idx, :]
    dis_map = np.sqrt(np.sum((map_idx-pt)*(map_idx-pt), 1))
    dis_map_idx = np.concatenate((dis_map.reshape(-1, 1), np.array(idx).reshape(-1, 1)), 1)
    sorted_dis = dis_map_idx[np.lexsort(dis_map_idx[:, ::-1].T)]
    return sorted_dis[:, 1].astype(int)


if __name__ == "__main__":

    # nice colors
    green = np.array([0.2980, 0.6, 0])
    darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    VermillionRed = np.array([156, 31, 46]) / 255

    ############# Set up the map and ground truth trajectory ####################
    # map of landmarks
    map = np.array([[1.3, 2.7],
                    [4.0, 2.6],
                    [0.1, 1.2],
                    [2.3, 1.1],
                    [1.7, 0.6],
                    [3.7, 3.1],
                    [2.4, 3.4],
                    [2.9, 2.0],
                    [1.2, 1.8]])
    # robot group truth positions
    x_gt = np.array([[0.0, 0.0],
                    [0.5, 0.4],
                    [1.0, 0.6],
                    [1.3, 1.1],
                    [1.7, 1.6],
                    [2.1, 1.7],
                    [2.4, 1.9],
                    [2.5, 2.4],
                    [2.7, 2.7],
                    [2.9, 3.0],
                    [3.1, 2.9],
                    [3.4, 2.5],
                    [3.5, 2.1],
                    [3.5, 1.7],
                    [3.3, 1.6],
                    [2.8, 1.6]])

    # plot the map and ground truth trajectory
    fig1 = plt.figure()
    plt.axis('equal')
    plt.plot(map[:, 0], map[:, 1], '+k', markersize=10, fillstyle='none')
    plt.plot(map[:, 0], map[:, 1], 'sk', markersize=10, fillstyle='none')
    line1 = plt.plot(x_gt[:, 0], x_gt[:, 1], '--', color=Darkgrey, alpha=0.7, linewidth=2, label='Ground truth')
    plt.plot(x_gt[:, 0], x_gt[:, 1], '.', color=Darkgrey, alpha=0.7, markersize=18)

    ########## Simulate noisy measurements in for of (dx, dy) = (x_landmark - x_robot, y_landmark - y_robot) ##########
    sigma_x = 0.05  # standard deviation along x
    sigma_y = 0.05  # standard deviation along y
    sigma_z = np.diag(np.power([sigma_x, sigma_y], 2))
    Lz = np.linalg.cholesky(sigma_z)

    z = {'landmark_coord': [], 'correspondences': []}  # measurements
    z_max = 4.5  # maximum sensor range in meters

    # create a kd-tree structure to search within the sensor range
    # mdlKDTree = KDTree(map)
    mdlKDTree = KDTree(map)
    map_gt = []
    for i in range(x_gt.shape[0]):
        # idx = mdlKDTree.query_radius(x_gt[i, :].reshape(-1, 1), z_max)
        idx = mdlKDTree.query_ball_point(x_gt[i, :], z_max)

        idx_sort = sort_idxlist(idx, map, x_gt[i, :])
        map_gt.append(map[idx_sort])

        z['landmark_coord'].append(map[idx_sort] - x_gt[i, :] + np.dot(Lz, np.random.randn(2, len(idx_sort))).T)
        z['correspondences'].append(np.array(idx_sort))
    map_gt = np.array(map_gt)
    z['landmark_coord'] = np.array(z['landmark_coord'])
    z['correspondences'] = np.array(z['correspondences'])

    ########## Simulate noisy odometry measurements ##########
    sigma_ux = 0.2
    sigma_uy = 0.2
    sigma_u = np.diag(np.power([sigma_ux, sigma_uy], 2))
    Lu = np.linalg.cholesky(sigma_u)

    u = np.diff(x_gt, axis=0) + np.dot(Lu, np.random.randn(2, x_gt.shape[0]-1)).T
    x_init = np.array([[0, 0]])
    sigma_init = np.array([[1e-3, 0], [0, 1e-3]])
    x_odom = np.vstack((x_init, np.cumsum(u, axis=0)))

    ########## Set up the linear system (Least Squares) ##########
    pose_dim = 2  # dimension of the robot position/pose
    landmark_dim = 2  # dimension of landmark
    pose_number = x_odom.shape[0]  # number of poses

    # Jacobians of the motion model
    # G = np.eye(2)

    # Jacobian of measurement model
    H = -np.eye(pose_dim)  # wrt robot position
    J = np.eye(landmark_dim)  # wrt landmark position

    # measurements must be filled based on correspondences
    # for new landmarks A has to be expanded
    # whereas for old ones the correspondences must be found
    seen_landmarks = []  # list of seen landmarks
    landmark_key = []

    # initialize EKF
    # combined state vector [robot position(2D); all landmarks(2D)]
    ekf = {}
    ekf['init_state'] = np.zeros([2+2*map.shape[0], 1])
    ekf['init_cov'] = block_diag(0.01 * np.eye(2), np.zeros([2*map.shape[0], 2*map.shape[0]]))
    ekf['predict_state'] = np.copy(ekf['init_state'])
    ekf['predict_cov'] = np.copy(ekf['init_cov'])
    ekf['state'] = np.copy(ekf['init_state'])
    ekf['cov'] = np.copy(ekf['init_cov'])
    ekf['state_dim'] = ekf['state'].shape[0]

    # for 95% ellipse drawing
    Chi2 = 5.991  # 2D
    phi = np.arange(-np.pi, np.pi+0.1, 0.1)
    circle = np.array([np.cos(phi), np.sin(phi)])
    scale = np.sqrt(Chi2)
    legend_plotting = True
    #### main loop ####
    for i in range(pose_number):
        # correction; we might have measurements at the initial position ##
        if len(z['correspondences'][i]):
            for j in range(len(z['correspondences'][i])):  # for each observed landmark j
                seen = any(landmark_idx == z['correspondences'][i][j] for landmark_idx in seen_landmarks)
                # if landmark j seen before, add a loop-closure
                if seen:
                    l_id = z['correspondences'][i][j]  # old landmark index
                    # assemble the Jacobian for the jth measurement
                    Hj = np.zeros([2, ekf['state_dim']])
                    Hj[:, 0:2] = H
                    Hj[:, 2+2*l_id:4+2*l_id] = J
                    # innovation covariance
                    S = np.dot(np.dot(Hj, ekf['predict_cov']), Hj.T) + sigma_z
                    # Kalman gain
                    K = np.dot(np.dot(ekf['predict_cov'], Hj.T), np.linalg.pinv(S))
                    # innovation
                    v = z['landmark_coord'][i][j, :].reshape(2, 1) - np.dot(Hj, ekf['predict_state'])
                    # apply correction
                    ekf['state'] = ekf['predict_state'] + np.dot(K, v)
                    ekf['cov'] = ekf['predict_cov'] - np.dot(np.dot(K, S), K.T)
                else:
                    # append landmark ID
                    seen_landmarks.append(z['correspondences'][i][j])
                    l_id = z['correspondences'][i][j]
                    ekf['state'][2 + 2 * l_id:4 + 2 * l_id] = ekf['state'][0:2] + z['landmark_coord'][i][j, :].reshape(
                        2, 1)
                    ekf['cov'][2 + 2 * l_id:4 + 2 * l_id, 2 + 2 * l_id:2 * l_id + 4] = ekf['cov'][0:2, 0:2] + sigma_z
                    ekf['cov'][2 + 2 * l_id:4 + 2 * l_id, 0:2] = ekf['cov'][0:2, 0:2]
                    ekf['cov'][0:2, 2 + 2 * l_id:4 + 2 * l_id] = ekf['cov'][0:2, 0:2]

        # ## prediction (propagation) ##
        # map
        ekf['predict_state'][2:] = np.copy(ekf['state'][2:])
        # robot position
        if i < pose_number-1:
            ekf['predict_state'][0:2] = ekf['state'][0:2].reshape(2, 1) + u[i, :].reshape(2, 1)
            ekf['predict_cov'] = ekf['cov'] + block_diag(sigma_u, np.zeros([2*map.shape[0], 2*map.shape[0]]))
        else:
            ekf['predict_state'][0:2] = np.copy(ekf['state'][0:2])
            ekf['predict_cov'] = np.copy(ekf['cov'])

        ## plotting ##
        for j in range(len(z['landmark_coord'][i])):
            plt.plot(np.hstack((x_gt[i, 0], map_gt[i][j, 0])), np.vstack((x_gt[i, 1], map_gt[i][j, 1])), '-',
                     color=green, alpha=0.3, linewidth=1.5)
        # plot odometry trajectory
        plt.plot(x_odom[i, 0], x_odom[i, 1], '.', color=VermillionRed, alpha=0.7, markersize=18)
        if i > 0:
            if legend_plotting:
                line2, = plt.plot(x_odom[i-1:i+1, 0], x_odom[i-1:i+1, 1], ':', color=VermillionRed, alpha=0.7,
                                  linewidth=2, label='Odometry')
                legend_plotting = False
            else:
                line2, = plt.plot(x_odom[i - 1:i + 1, 0], x_odom[i - 1:i + 1, 1], ':', color=VermillionRed, alpha=0.7,
                                  linewidth=2)

        if legend_plotting:
            line3, = plt.plot(ekf['state'][0], ekf['state'][1], '.', color=darkblue, alpha=0.7, markersize=22,
                              label='EKF-Robot')
        else:
            line3, = plt.plot(ekf['state'][0], ekf['state'][1], '.', color=darkblue, alpha=0.7, markersize=22)

        if i > 0:
            ellipse = np.dot(np.sqrt(Chi2) * np.linalg.cholesky(ekf['cov'][0:2, 0:2]), circle) \
                      + ekf['state'][0:2].reshape(2, 1)
            plt.plot(ellipse[0, :], ellipse[1, :], color=darkblue, linewidth=1)

        if positiveCheck(ekf['init_cov'][0:2, 0:2]):
            ellipse = np.dot(np.sqrt(Chi2) * np.linalg.cholesky(ekf['init_cov'][0:2, 0:2]), circle) \
                      + ekf['init_state'][0:2].reshape(2, 1)
            plt.plot(ellipse[0, :], ellipse[1, :], color=darkblue, linewidth=1)
        plt.pause(0.5)

    # SLAM map
    line4, = plt.plot(ekf['state'][2::2], ekf['state'][3::2], '*', color=darkblue, alpha=0.9, markersize=14, label='Locaton')
    # draw 95% uncertainty ellipse
    for i in range(map.shape[0]):
        ellipse = np.dot(np.sqrt(Chi2) * np.linalg.cholesky(ekf['cov'][2+2*i:4+2*i, 2+2*i:4+2*i]), circle) \
                  + ekf['state'][2+2*i:4+2*i].reshape(2, 1)
        plt.plot(ellipse[0, :], ellipse[1, :], color=darkblue, linewidth=1)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    fig2 = plt.subplots()
    ax2 =plt.gca()
    ax2.spy(abs((ekf['cov'] + ekf['cov']) / 2) > np.finfo(float).eps, markersize=5, color=Darkgrey)
    ax2.set_title(r'Covariance Matrix $\Sigma$')
    plt.show()

    fig3 = plt.subplots()
    ax3 = plt.gca()
    ax3.spy(np.float32(np.linalg.inv(abs((ekf['cov'] + ekf['cov'].T) / 2))) > np.finfo(float).eps, markersize=5, color=Darkgrey)
    ax3.set_title(r'Information Matrix $\Sigma^{-1}$')
    plt.show()

