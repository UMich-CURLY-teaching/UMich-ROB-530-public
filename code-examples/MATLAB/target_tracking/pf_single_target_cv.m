% Toy example for tracking a single target using a particle filter
% and range-bearing measurements.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   01/22/2019

clc; clear; close all

% First simulate a target that moves on a curved path; we assume ownship is at
% the origin (0,0) and received noisy range and bearing measurements of the
% target location. There is no knowledge of the target motion, but, we
% assume a constant velocity random walk motion model.

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
   % sample from a zero mean Gaussian with covariance R
   noise = L * randn(2,1); 
   z(:,i) = [sqrt(gt.x(i)^2 + gt.y(i)^2); atan2(gt.x(i), gt.y(i))] + noise;
end

% build the system 
sys = [];
dt = 1;
F = [1 0 dt 0;
    0 1 0 dt;
    0 0 1 0;
    0 0 0 1];
sys.f = @(x,w) F * x + w;
sys.h = @(x)  [sqrt(x(1)^2 + x(2)^2); atan2(x(1), x(2))];
sys.Q = diag([1e-1, 1e-1, 1e-2, 1e-2]);
sys.R = diag([0.05^2; 0.01^2]);

% initialization! 
init = [];
init.n = 1000;
init.x(1,1) = z(1,1) * sin(z(2,1));
init.x(2,1) = z(1,1) * cos(z(2,1));
init.x(3,1) = 0;
init.x(4,1) = 0;
init.Sigma = 1 * eye(4);

filter = particle_filter(sys, init);
x = nan(4,1);     % state

% incremental visualization
green = [0.2980 .6 0];
% crimson = [220,20,60]/255; 
% darkblue = [0 .2 .4];
% Darkgrey = [.25 .25 .25];
% darkgrey = [.35 .35 .35];
% lightgrey = [.7 .7 .7];
% Lightgrey = [.9 .9 .9];
% VermillionRed = [156,31,46]/255;
% DupontGray = [144,131,118]/255;
% Azure = [53, 112, 188]/255;
% purple = [178, 102, 255]/255;
% orange = [255,110,0]/255;

fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

h = figure; hold on
plot(0,0, '^', 'MarkerFaceColor', 'b', 'markersize', 20)
plot(gt.x, gt.y, '-', 'linewidth', 2)
grid on, axis auto equal, axis([-6 6 -1 5])
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
hp = plot(filter.p.x(1,:), filter.p.x(2,:),'.','Color', [green, .25]);

% main loop; iterate over the measurements
for i = 2:size(z,2)
    filter.sample_motion_cv();
    filter.importance_measurement(z(:,i));
    if filter.Neff < filter.n/5
        filter.resampling();
    end
    wtot = sum(filter.p.w);
    if wtot > 0
        x(1,i) = sum(filter.p.x(1,:)' .* filter.p.w) / wtot;
        x(2,i) = sum(filter.p.x(2,:)' .* filter.p.w) / wtot;
        x(3,i) = sum(filter.p.x(3,:)' .* filter.p.w) / wtot;
        x(4,i) = sum(filter.p.x(4,:)' .* filter.p.w) / wtot;
    else
        warning('Total weight is zero or nan!')
        disp(wtot)
        x(:,i) = nan(2,1);
    end
    % show particles
    set(hp,'XData',filter.p.x(1,:),'YData', filter.p.x(2,:));
    drawnow limitrate
    pause(0.05)
end
close(h)

% plotting
figure; hold on, grid on
plot(0,0, '^', 'MarkerFaceColor', 'b', 'markersize', 20)
plot(gt.x, gt.y, '-', 'linewidth', 2)
plot(x(1,:), x(2,:), '-k', 'linewidth', 2)
legend('ownship', 'ground truth', 'PF', 'location', 'best')
grid on, axis auto equal tight
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
figuresize(21,21,'cm')
% print -opengl -dpng -r600 pf_example_tracking_cv.png

figure; hold on, grid on
plot(x(3,:), 'linewidth', 2)
plot(x(4,:), 'linewidth', 2)
legend('velocity - $x_1$', 'velocity - $x_2$', 'location', 'best', 'Interpreter','latex')
grid on, axis auto tight
xlabel('time', 'fontsize', fsize, 'Interpreter','latex')
ylabel('velocity', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
figuresize(21,14,'cm')
% print -opengl -dpng -r600 pf_example_tracking_vel.png
