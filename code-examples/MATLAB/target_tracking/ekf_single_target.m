% Toy example for tracking a single target using an Extended Kalman filter
% and range-bearing measurements.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   11/25/2018

clc; clear; close all

% First simulate a target that moves on a curved path; we assume ownship is at
% the origin (0,0) and received noisy range and bearing measurements of the
% target location. There is no knowledge of the target motion, hence, we
% assume a random walk motion model.

% ground truth data
gt = [];
gt.x = -5:0.1:5;
gt.y = 1 * sin(gt.x) + 3;

% measurements
R = diag([0.05^2; 0.01^2]);
% Cholesky factor of covariance for sampling
L = chol(R, 'lower');
z = [];
for i = 1:length(gt.x)
   % sample from a zero mean Gaussian with covariance V
   noise = L * randn(2,1); 
   z(:,i) = [sqrt(gt.x(i)^2 + gt.y(i)^2); atan2(gt.x(i), gt.y(i))] + noise;
end

% build the system 
sys = [];
sys.A = eye(2);
sys.B = [];
sys.f = @(x) [x(1); x(2)];
sys.H = @(x) [x(1)/(x(1)^2 + x(2)^2)^(1/2), x(2)/(x(1)^2 + x(2)^2)^(1/2);
              x(2)/(x(1)^2 + x(2)^2), -x(1)/(x(1)^2 + x(2)^2)];
sys.h = @(x)  [sqrt(x(1)^2 + x(2)^2); atan2(x(1), x(2))];
sys.Q = 1e-3 * eye(2);
sys.R = diag([0.05^2; 0.01^2]);

% initilize the state using the first measurement
init = [];
init.x = zeros(2,1);
init.x(1,1) = z(1,1) * sin(z(2,1));
init.x(2,1) = z(1,1) * cos(z(2,1));
init.Sigma = 1 * eye(2);

filter = ekf(sys, init);
x = init.x;     % state
% main loop; iterate over the measurements
for i = 2:size(z,2)
    filter.prediction();
    filter.correction(z(:,i));
    x(:,i) = filter.x;
end

% plotting
fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

figure; hold on
plot(0,0, '^', 'MarkerFaceColor', 'b', 'markersize', 20)
plot(gt.x, gt.y, '-', 'linewidth', 2)
plot(x(1,:), x(2,:), '-k', 'linewidth', 2)
legend('ownship', 'ground truth', 'EKF', 'location', 'best')
grid on, axis auto equal tight
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
figuresize(21,21,'cm')
% print -opengl -dpng -r600 ekf_example_tracking.png
