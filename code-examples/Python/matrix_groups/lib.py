#!usr/bin/env python

#
# Rotation Matrix
# Author: Fangtong Liu
# Date: 05/14/2020
#

import numpy as np


def R2d(theta):
    # 2D rotation matrix
    # Input:
    #   theta:  angle is in radian
    # Output:
    #   R:  2D Rotation matrix

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R


def Rx(roll):
    # Rotation matrix about X axis (roll)
    # Input:
    #   roll:   angle in radian about X axis
    # Output:
    #   H:  3D rotation matrix
    sr = np.sin(roll)
    cr = np.cos(roll)

    H = np.array([[1, 0, 0],
                  [0, cr, -sr],
                  [0, sr, cr]])
    return H


def Ry(pitch):
    # Rotation matrix about Y axis(pitch)
    # Input:
    #   pitch:   angle in radian about Y axis
    # Output:
    #   H:  3D rotation matrix
    sp = np.sin(pitch)
    cp = np.cos(pitch)

    H = np.array([[cp, 0, sp],
                  [0, 1, 0],
                  [-sp, 0, cp]])

    return H


def Rz(yaw):
    # Rotation matrix about Z axis(yaw)
    # Input:
    #   yaw:   angle in radian about Z axis
    # Output:
    #   H:  3D rotation matrix
    sy = np.sin(yaw)
    cy = np.cos(yaw)

    H = np.array([[cy, -sy, 0],
                  [sy, cy, 0],
                  [0, 0, 1]])

    return H


def Rzyx(yaw, pitch, roll):
    # Rotation matrix for yaw pitch roll angles
    # Order of rotation R = Rz * Ry * Rx
    # Extrinsic rotations; The XYZ system rotates, while xyz is fixed
    # Verifeid using MATLAB rotm = eul2rotm([yaw, pitch, roll], 'ZYX')
    # Input:
    #   yaw, pitch, roll:   rotation angles (radian) about Z, Y, X
    # Output:
    #   R: 3D rotation matrix
    R = np.dot(np.dot(Rz(yaw), Ry(pitch)), Rx(roll))
    return R


def R2ypr(R):
    # Extracts angles from a rotation matrix
    # yaw(Z) pitch(Y), roll(X) angles from rotation matrix
    # Extrinsic rotations; The XYZ system rotates, while xyz is fixed
    # verified using MATLAB eul = rotm2eul(rotm, 'ZYX')
    # Input:
    #   R:  3D rotation matrix
    # Output:
    #   ypr:    yaw, pitch, roll

    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], R[0, 0] * np.cos(yaw) + R[1, 0] * np.sin(yaw))
    roll = np.arctan2(R[2, 1], R[2, 2])

    return np.array([yaw, pitch, roll])

