#!usr/bin/env python

#
# Incremental least squares SLAM
# Author: Fangtong Liu
# Date: 04/29/2020
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.linalg import block_diag
from cholupdate_function import cholupdate


# This function is used to re-arrange the index found by KD-Tree
# with an increment of distance from robot to landmarks
def sort_idxlist(idx, map, pt):
    # Input:
    #   idx:    array of nearest landmarks indices within z_max
    #   map:    array of landmarks
    #   pt:     ground truth point which is used to search for nearest landmarks
    # Output:
    #   sorted_dist:    sorted index with increment of distances from robot to landmarks
    map_idx = map[idx, :]
    dis_map = np.sqrt(np.sum((map_idx-pt)*(map_idx-pt), 1))
    dis_map_idx = np.concatenate((dis_map.reshape(-1, 1), np.array(idx).reshape(-1, 1)), 1)
    sorted_dis = dis_map_idx[np.lexsort(dis_map_idx[:, ::-1].T)]
    return sorted_dis[:, 1].astype(int)


if __name__ == "__main__":

    # nice colors
    green = np.array([0.298, 0.6, 0])
    crimson = np.array([220, 20, 60]) / 255
    darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    darkgrey = np.array([0.35, 0.35, 0.35])
    lightgrey = np.array([0.7, 0.7, 0.7])
    Lightgrey = np.array([0.9, 0.9, 0.9])
    VermillionRed = np.array([156, 31, 46]) / 255
    DupontGray = np.array([144, 131, 118]) / 255
    Azure = np.array([53, 112, 255]) / 255
    purple = np.array([178, 102, 255]) / 255
    orange = np.array([255, 110, 0]) / 255

    ####### Setup the map and ground truth trajectory #######
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
    plt.grid(True)
    plt.axis("auto")
    plt.plot(map[:, 0], map[:, 1], '+k', markersize=10, fillstyle='none')
    plt.plot(map[:, 0], map[:, 1], 'sk', markersize=10, fillstyle="none")
    plt.plot(x_gt[:, 0], x_gt[:, 1], '--', color=Darkgrey, alpha=0.7, linewidth=2, label='Ground truth')
    plt.plot(x_gt[:, 0], x_gt[:, 1], '.', color=Darkgrey, alpha=0.7, markersize=18)
    legend_plotting = True

    ###### Simulate noisy measurement in for of (dx, dy) = (x_landmark - x_robot, y_landmark - y_robot) ######
    sigma_x = 0.05  # standard deviation along x
    sigma_y = 0.05  # standard deviation along y
    sigma_z = np.diag(np.power([sigma_x, sigma_y], 2))
    Lz = np.linalg.cholesky(sigma_z)

    z = {'landmark_coord': [], 'correspondence': []}  # measurements
    z_max = 1.5  # maximum sensor range in meters
    # create a kd-tree structure to search within the sensor range
    mdlKDTree = KDTree(map)
    map_gt = []
    for i in range(x_gt.shape[0]):
        idx = mdlKDTree.query_ball_point(x_gt[i, :], z_max)
        idx_sorted = sort_idxlist(idx, map, x_gt[i, :])

        map_gt_i = map[idx_sorted, :]  # observed landmarks position at time t
        map_gt.append(map_gt_i)
        z['landmark_coord'].append(map_gt_i - x_gt[i, :] + np.dot(Lz, np.random.randn(2, len(idx_sorted))).T)
        z['correspondence'].append(np.array(idx_sorted))

    map_gt = np.array(map_gt)

    ###### Simulate noisy odometry measurements ######
    sigma_ux = 0.2
    sigma_uy = 0.2
    sigma_u = np.diag(np.power([sigma_ux, sigma_uy], 2))
    Lu = np.linalg.cholesky(sigma_u)
    u = np.diff(x_gt, axis=0) + np.dot(Lu, np.random.randn(2, x_gt.shape[0]-1)).T
    x_init = np.array([0, 0])
    sigma_init = 1e-3 * np.eye(2)
    x_odom = np.vstack((x_init, np.cumsum(u, axis=0)))

    ###### Setup the linear system (Least Squares) ######
    nr = 2  # dimension of the robot position/pose
    nl = 2  # dimension of landmark
    pose_number = x_odom.shape[0]  # number of poses

    # Jacobians of the motion model
    F = -np.eye(2)
    G = np.eye(2)
    # Jacobians of the measurement model
    H = -np.eye(2)  # wrt robot position
    J = np.eye(2)  # wrt landmark position

    # measurements must be filled based on correspondences
    # for new landmarks A has to be expanded
    # whereas for old ones the correspondences must be found
    seen_landmarks = []  # list of seen landmarks
    landmark_key = []
    pose_key = []
    ###### main loop ######
    for i in range(pose_number):
        if i == 0:
            # initialize b (right hand side)
            b = np.dot(np.linalg.inv(Lu), x_init).reshape(-1, 1)
            # initialize A using initial condition
            A = np.dot(np.linalg.inv(sigma_init), G)
            pose_key.append(0)
            # initialize Cholesky factor R = L.T
            R = np.linalg.cholesky(np.dot(A.T, A)).T
            print(1)
        else:
            # Compute b (right hand side)
            b = np.vstack((b, np.dot(np.linalg.inv(Lu), u[i-1, :]).reshape(-1, 1)))
            # constructing A using robot motion
            A = block_diag(A, np.dot(np.linalg.inv(Lu), G))
            A[-nr:, pose_key[i-1]:pose_key[i-1]+2] = np.dot(np.linalg.inv(Lu), F)
            # update pose key
            pose_key.append(A.shape[1]-nr)
            # expand Cholesky factor
            R = block_diag(R, np.zeros(J.shape))
            for k in range(nl, 0, -1):
                x = np.copy(A[-k, :].T)
                R = cholupdate(R, x)

        if len(z['correspondence'][i]):
            for j in range(len(z['correspondence'][i])):
                # check whether current landmark is seen before
                seen = any(landmark_idx == z['correspondence'][i][j] for landmark_idx in seen_landmarks)
                # old landmarks, add a loop-closure
                if seen:
                    nA, mA = A.shape
                    Asub = np.zeros([nl, mA])  # new submatrix in the bottom of A
                    # find the landmark id stored in A
                    l_id = landmark_key[seen_landmarks.index(z['correspondence'][i][j])]
                    # add Jacobian of the corresponding landmark and robot position/pose
                    Asub[:, l_id-1:l_id+nl-1] = np.dot(np.linalg.inv(Lz), J)
                    Asub[:, pose_key[i]:pose_key[i]+2] = np.dot(np.linalg.inv(Lz), H)
                    # append A and b
                    A = np.vstack((A, Asub))
                    b = np.vstack((b, np.dot(np.linalg.inv(Lz), z['landmark_coord'][i][j, :]).reshape(-1, 1)))
                    # expand Cholesky factor
                    for k in range(nr, 0, -1):
                        x = np.copy(A[-k, :].T)
                        R = cholupdate(R, x)
                # new landmark, exploration; expand the state and the Jacobian
                else:
                    # append landmark id
                    seen_landmarks.append(z['correspondence'][i][j])
                    # expand A based on J
                    A = block_diag(A, np.dot(np.linalg.inv(Lz), J))
                    # Jacobian of the corresponding robot position/pose
                    A[-H.shape[0]:, pose_key[i]:pose_key[i]+2] = np.dot(np.linalg.inv(Lz), H)
                    b = np.vstack((b, np.dot(np.linalg.inv(Lz), z['landmark_coord'][i][j, :]).reshape(-1, 1)))
                    # update landmark key
                    landmark_key.append(A.shape[1]-1)
                    # expand Cholesky factor
                    R = block_diag(R, np.zeros(J.shape))
                    for k in range(nl, 0, -1):
                        x = np.copy(A[-k, :].T)
                        R = cholupdate(R, x)

        ###### plotting ######
        # plot ground truth pose and landmarks
        for j in range(len(z['correspondence'][i])):
            plt.plot(np.hstack((x_gt[i, 0], map_gt[i][j, 0])), np.vstack((x_gt[i, 1], map_gt[i][j, 1])), '-',
                     color=green, alpha=0.3, linewidth=1.5)
        # plot odometry trajectory
        plt.plot(x_odom[i, 0], x_odom[i, 1], '.', color=VermillionRed, alpha=0.7, markersize=18)
        if i > 0:
            if legend_plotting:
                plt.plot(x_odom[i-1:i+1, 0], x_odom[i-1:i+1, 1], ':', color=VermillionRed, alpha=0.7,
                         linewidth=2, label='Odometry')
                legend_plotting = False
            else:
                plt.plot(x_odom[i - 1:i + 1, 0], x_odom[i - 1:i + 1, 1], ':', color=VermillionRed, alpha=0.7,
                         linewidth=2)

        # plt.pause(0.05)

    x_slam = np.dot(np.linalg.inv(R), np.dot(np.linalg.inv(R.T), np.dot(A.T, b)))
    # Final SLAM trajectory
    pose_key = np.array(pose_key)
    plt.plot(x_slam[pose_key], x_slam[pose_key+1], '-', color=darkblue, alpha=0.7, linewidth=2, label='SLAM')
    plt.plot(x_slam[pose_key], x_slam[pose_key + 1], '.', color=darkblue, alpha=0.7, markersize=18)
    # SLAM map
    landmark_key = np.array(landmark_key)
    plt.plot(x_slam[landmark_key-1], x_slam[landmark_key], '*', color=darkblue, alpha=0.7, markersize=14)
    plt.legend(loc='best')
    plt.show()

    # Solve using QR factorization
    Q, R = np.linalg.qr(A)
    x_qr = np.dot(np.dot(np.linalg.inv(R), Q.T), b)

    # Solve using Cholesky factorization
    L = np.linalg.cholesky(np.dot(A.T, A))
    x_chol = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L), np.dot(A.T, b)))

    print(r'norm(A \ b - R \ Q^T * b) = ', np.linalg.norm(x_slam - x_qr))
    print(r'norm(A \ b - L^T \ (L \ A^T * b)) = ', np.linalg.norm(x_slam - x_chol))


    # spy matrix
    fig2 = plt.figure(figsize=(12, 12))

    plt.subplot(221)
    ax1 = plt.gca()
    ax1.set_title(r'$A$')
    ax1.spy(A, markersize=5, color=Darkgrey, origin='lower', aspect='equal')
    label1 = 'nz = ' + str(len(np.nonzero(A.reshape(-1))[0]))
    ax1.set_xlabel(label1, fontweight='bold')

    plt.subplot(222)
    ax2 = plt.gca()
    ax2.set_title(r'$A^\mathsf{T}A$')
    ax2.spy(np.dot(A.T, A), markersize=5, color=Darkgrey)
    label2 = 'nz = ' + str(len(np.nonzero(np.dot(A.T, A).reshape(-1))[0]))
    ax2.set_xlabel(label2, fontweight='bold')

    plt.subplot(223)
    ax3 = plt.gca()
    ax3.set_title(r'$\mathsf{R}-QR$')
    ax3.spy(R[0:A.shape[1], :], markersize=5, color=darkblue)
    label3 = 'nz = ' + str(len(np.nonzero(R[0:A.shape[1], :].reshape(-1))[0]))
    ax3.set_xlabel(label3, fontweight='bold')

    plt.subplot(224)
    ax4 = plt.gca()
    ax4.set_title(r'$\mathsf{R}-Cholesky$')
    ax4.spy(L.T, markersize=5, color=VermillionRed)
    label4 = 'nz = ' + str(len(np.nonzero(L.reshape(-1))[0]))
    ax4.set_xlabel(label4, fontweight='bold')
    plt.show()








