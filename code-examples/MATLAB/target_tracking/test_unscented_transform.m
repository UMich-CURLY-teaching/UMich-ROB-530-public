% Test script for unscented transform class
%
%   Author: Maani Ghaffari Jadidi
%   Date:   01/17/2019

clc; clear; close all

% creat a random mean and covariance
n = 3;
x = randn(3,1);
P = randn(3); P = P*P';

% build a linear model
A = randn(2,3);
f = @(z) A*z;

% propagate the uncertainty using UT and affine model to compare
kappa = 2;
ut = unscented_transform(x, P, f, kappa);
ut.propagate;

disp(['kappa = ', num2str(kappa)])
disp(['norm(ut.mean - A*x)    = ',  num2str(norm(ut.mean - A*x))])
disp(['norm(ut.Cov - A*P*A^T) = ',  num2str(norm(ut.Cov - A*P*A', 'fro'))])

kappa = 1;
ut.set(x, P, f, kappa);
ut.propagate;

disp('--------')
disp(['kappa = ', num2str(kappa)])
disp(['norm(ut.mean - A*x)    = ',  num2str(norm(ut.mean - A*x))])
disp(['norm(ut.Cov - A*P*A^T) = ',  num2str(norm(ut.Cov - A*P*A', 'fro'))])

kappa = 0;
ut.set(x, P, f, kappa);
ut.propagate;

disp('--------')
disp(['kappa = ', num2str(kappa)])
disp(['norm(ut.mean - A*x)    = ',  num2str(norm(ut.mean - A*x))])
disp(['norm(ut.Cov - A*P*A^T) = ',  num2str(norm(ut.Cov - A*P*A', 'fro'))])