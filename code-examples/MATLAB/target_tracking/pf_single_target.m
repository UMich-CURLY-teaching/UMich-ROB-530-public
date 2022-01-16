% Toy example for tracking a single target using a particle filter
% and range-bearing measurements.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   01/22/2019

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
   % sample from a zero mean Gaussian with covariance R
   noise = L * randn(2,1); 
   z(:,i) = [sqrt(gt.x(i)^2 + gt.y(i)^2); atan2(gt.x(i), gt.y(i))] + noise;
end

% build the system 
sys = [];
sys.f = @(x,w) [x(1); x(2)] + w;
sys.h = @(x)  [sqrt(x(1)^2 + x(2)^2); atan2(x(1), x(2))];
sys.Q = 1e-1 * eye(2);
sys.R = diag([0.05^2; 0.01^2]);

% initialization! 
init = [];
init.n = 100;
init.x(1,1) = z(1,1) * sin(z(2,1));
init.x(2,1) = z(1,1) * cos(z(2,1));
init.Sigma = 1 * eye(2);

filter = particle_filter(sys, init);
x = nan(2,1);     % state

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
    filter.sample_motion();
    filter.importance_measurement(z(:,i));
    if filter.Neff < filter.n/5
        filter.resampling();
    end
    wtot = sum(filter.p.w);
    if wtot > 0
        x(1,i) = sum(filter.p.x(1,:)' .* filter.p.w) / wtot;
        x(2,i) = sum(filter.p.x(2,:)' .* filter.p.w) / wtot;
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
% print -opengl -dpng -r600 pf_example_tracking.png
