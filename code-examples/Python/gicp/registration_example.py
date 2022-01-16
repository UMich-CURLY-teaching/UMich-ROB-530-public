#!usr/bin/env python

#
# 3-D Point Cloud Registration and Stitching example of GICP
#
# Author: Fangtong Liu
# Date: 06/03/2020
#

import numpy as np
import open3d as o3d
from gicp_SE3 import gicp_SE3
from gicp_Sim3 import gicp_Sim3
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


if __name__ == "__main__":
    ptCloudRef = o3d.io.read_point_cloud("ptCloud/livingRoomData1.ply")
    ptCloudCurrent = o3d.io.read_point_cloud("ptCloud/livingRoomData2.ply")

    gridSize = 0.1
    fixed = ptCloudRef.voxel_down_sample(voxel_size=gridSize)
    moving = ptCloudCurrent.voxel_down_sample(voxel_size=gridSize)

    ################# solve for tf ##########################
    T = gicp_SE3(np.asarray(fixed.points), np.asarray(moving.points))
    # T = gicp_Sim3(np.asarray(fixed.points), np.asarray(moving.points))
    accumTform = np.copy(T)

    ptCloudAligned = ptCloudCurrent.transform(T)

    ptCloudScene = o3d.geometry.PointCloud()
    ptCloudScene += ptCloudAligned
    ptCloudScene += ptCloudRef

    ###### Visualize the input images #########3
    fig1 = plt.figure()
    ax1 = plt.subplot(221)
    color = loadmat("ptCloud/ptCloudColor1.mat")
    colorData = color['ptCloudColor1']
    plt.imshow(colorData)
    ax1.set_title('First input image', fontsize=12)

    ax2 = plt.subplot(223)
    color = loadmat("ptCloud/ptCloudColor2.mat")
    colorData = color['ptCloudColor2']
    plt.imshow(colorData)
    ax2.set_title('Second input image', fontsize=12)

    ax3 = fig1.add_subplot(1, 2, 2, projection='3d')
    ax3.scatter(np.asarray(ptCloudScene.points)[:, 0], np.asarray(ptCloudScene.points)[:, 1],
                np.asarray(ptCloudScene.points)[:, 2], c=np.asarray(ptCloudScene.colors), s=5)
    ax3.view_init(-80, -80)
    ax3.set_title('Initial world scene', fontsize=12)
    ax3.set_xlabel(r'X(m)')
    ax3.set_ylabel(r'Y(m)')
    ax3.set_zlabel(r'Z(m)')

    plt.show()

    # --------------------------------------------------------------------
    # Store the transformation object that accumulates the transformation
    # --------------------------------------------------------------------

    # fig2 = plt.figure(figsize=(12, 12))
    # ax = plt.gca(projection='3d')
    # ptCloudScene = ptCloudScene.voxel_down_sample(voxel_size=0.01)
    # ax.scatter(np.asarray(ptCloudScene.points)[:, 0], np.asarray(ptCloudScene.points)[:, 1],
    #            np.asarray(ptCloudScene.points)[:, 2], c=np.asarray(ptCloudScene.colors), s=5)
    # ax.view_init(-80, -80)
    # ax.set_title('Updated world scene', fontsize=12)
    # ax.set_xlabel(r'X(m)')
    # ax.set_ylabel(r'Y(m)')
    # ax.set_zlabel(r'Z(m)')

    for i in tqdm(range(2, 44)):
        ptCloudCurrent = o3d.io.read_point_cloud("ptCloud/livingRoomData%d.ply" % i)

        # Use previous moving point cloud as reference
        fixed = moving
        moving = ptCloudCurrent.voxel_down_sample(voxel_size=gridSize)

        ######################## solve for tf #################################
        T = gicp_SE3(np.asarray(fixed.points), np.asarray(moving.points))
        # T = gicp_Sim3(np.asarray(fixed.points), np.asarray(moving.points))

        # Transform the current point cloud to the reference coordinate system
        # defined by the first point cloud
        accumTform = np.dot(T, accumTform)
        ptCloudAligned = ptCloudCurrent.transform(accumTform)
        # ptCloudAligned = ptCloudAligned.voxel_down_sample(voxel_size=0.01)
        ptCloudScene += ptCloudAligned

        # Visualize the world scene
        # ax.scatter(np.asarray(ptCloudAligned.points)[:, 0], np.asarray(ptCloudAligned.points)[:, 1],
        #            np.asarray(ptCloudAligned.points)[:, 2], c=np.asarray(ptCloudAligned.colors), s=5)
        # ax.view_init(-80, -80)
        # ax.set_title('Updated world scene', fontsize=12)

        # plt.pause(0.05)

    fig2 = plt.figure(figsize=(12, 12))
    ax = plt.gca(projection='3d')
    ax.scatter(np.asarray(ptCloudScene.points)[:, 0], np.asarray(ptCloudScene.points)[:, 1],
               np.asarray(ptCloudScene.points)[:, 2], c=np.asarray(ptCloudScene.colors), s=5)
    ax.view_init(-80, -80)
    ax.set_title('Updated world scene', fontsize=12)
    ax.set_xlabel(r'X(m)')
    ax.set_ylabel(r'Y(m)')
    ax.set_zlabel(r'Z(m)')
    o3d.io.write_point_cloud("ptCloudScene_SE3.ply", ptCloudScene, write_ascii=True)
    plt.show()
