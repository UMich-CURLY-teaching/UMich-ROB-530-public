#!usr/bin/env python

#
# colamd used by a c file
# Author: Fangtong Liu
# Date: 04/05/2020
#

from ctypes import *

so_file = '/home/m/repos/mobile-robotics-code/python/slam/colamd.so'
colamd_cfile = CDLL(so_file)


def COLAMD_RECOMMENDED(nnz, n_row, n_col):
    # Returns recommended value of Alen for use by colamd.  Returns -1
    # if any input argument is negative.
    # Input:
    #   nnz:	Number of nonzeros in the matrix A.  This must
    # 			be the same value as p [n_col] in the call to
    # 			colamd
    #   n_row:  Number of rows in the matrix A.
    #   n_col:	Number of columns in the matrix A.

    c_return = colamd_cfile.COLAMD_RECOMMENDED(nnz, n_row, n_col)
    if c_return != -1:
        return c_return
    else:
        return "C Function failed, check inputs"

def colamd (n_row, n_col, Alen, A, p, knobs[COLAMD_KNOBS], stats[COLAMD_STATS]):
    # Computes a column ordering (Q) of A such that P(AQ)=LU or (AQ)'AQ=LL'
    # have less fill-in and require fewer floating point operations than
    # factorizing the unpermuted matrix A or A'A, respectively.
    # Input:
    #   n_row:  Number of rows in the matrix A.
    #   n_col:  Number of columns in the matrix A.
    #   Alen:   Alen >= 2*nnz + 6*(n_col+1) + 4*(n_row+1) + n_col
    #   A:      A is an integer array of size Alen.
    #   p:      p is an integer array of size n_col+1.
    #

