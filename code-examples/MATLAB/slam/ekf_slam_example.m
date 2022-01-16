%{  
    Copyright (C) 2018  Maani Ghaffari Jadidi
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details. 
%}

% EKF SLAM example using known correspondences and linear motion and
% measurement models.

clc; clear; close all

fsize = 16; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

% nice colours
green = [0.2980 .6 0];
darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
VermillionRed = [156,31,46]/255;

%% Setup the map and ground truth trajectory
% map of landmarks
map = [1.3, 2.7
       4.0, 2.6
       0.1, 1.2
       2.3, 1.1
       1.7, 0.6
       3.7, 3.1
       2.4, 3.4
       2.9, 2.0
       1.2, 1.8];
% robot group truth positions
x_gt = [0,0; 
        0.5, 0.4; 
        1, 0.6; 
        1.3, 1.1;
        1.7, 1.6;
        2.1, 1.7;
        2.4, 1.9
        2.5, 2.4
        2.7, 2.7
        2.9, 3.0
        3.1, 2.9
        3.4, 2.5
        3.5, 2.1
        3.5, 1.7
        3.3, 1.6
        2.8, 1.6];
    

% plot the map and ground truth trajectory
h_leg = []; % legend handle
figure; 
axis auto equal
figuresize(21,21,'cm')
hold on; grid on; 
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
plot(map(:,1), map(:,2), '*k', 'MarkerSize', 10)
plot(map(:,1), map(:,2), 'sk', 'MarkerSize', 10)
h_leg{1} = plot(x_gt(:,1), x_gt(:,2), '--', 'color', [Darkgrey, 0.7], 'linewidth', 2);
plot(x_gt(:,1), x_gt(:,2), '.', 'color', [Darkgrey, 0.7], 'MarkerSize', 18)

%% Simulate noisy measuremest in for of (dx, dy) = (x_landmark - x_robot, y_landmark - y_robot)
sigma_x = 0.05;  % standard deviation along x
sigma_y = 0.05;  % standard deviation along y
Sigma_z = blkdiag(sigma_x^2, sigma_y^2);
Lz = chol(Sigma_z, 'lower');

z = cell(size(x_gt,1),2);         % measurements
z_max = 4.5;      % maximum sensor range in meters
% create a kd-tree structure to search within the sensor range
MdlKDT = KDTreeSearcher(map);
map_gt = [];
for i = 1:size(x_gt,1)
    Idx = rangesearch(MdlKDT, x_gt(i,:), z_max);
    map_gt{i,1} = map(Idx{1},:);
    z{i,1} = map_gt{i,1} - x_gt(i,:) + (Lz * randn(size(map_gt{i,1},1),2)')'; % landmark coordinates
    z{i,2} = Idx{1};        % correspondences
end

%% Simulate noisy odometry measurements
sigma_ux = 0.2;
sigma_uy = 0.2;
Sigma_u = blkdiag(sigma_ux^2, sigma_uy^2);
Lu = chol(Sigma_u, 'lower');

u = diff(x_gt) + (Lu * randn(size(x_gt,1)-1,2)')';
x_init = [0, 0];
Sigma_init = [1e-3, 0; 0, 1e-3];
x_odom = [x_init; cumsum(u)];

%% Set up the linear system (Least Squares)
% nr = 2; % dimension of the robot position/pose
% nl = 2; % dimension of landmark
np = size(x_odom,1); % number of poses
% Jacobians of the motion model
% G = eye(2);
% Jacobian of measurement model
H = -eye(2);     % wrt to robot position
J = eye(2);      % wrt to landmark position

% measurements must be filled based on correspondences
% for new landmarks A has to be expanded whereas for old ones the
% correspondences must be found.
seen_landmarks = [];    % list of seen landmarks
landmark_key = [];

% initialize EKF
% robot position (2D) + all landmarks (2D)
ekf = [];
ekf.init_state = [zeros(2,1); zeros(2*size(map,1),1)];
ekf.init_cov = blkdiag(.01*eye(2,2), zeros(2*size(map,1)));
ekf.predict_state = ekf.init_state;
ekf.predict_cov = ekf.init_cov;
ekf.state = ekf.init_state;
ekf.cov = ekf.init_cov;
ekf.state_dim = size(ekf.state,1);

% for 95% ellipse drawing
Chi2 = 5.991; % 2D
phi = (-pi:.01:pi)';
circle = [cos(phi), sin(phi)];
scale = sqrt(5.991);
% ellipse = (sqrt(Chi2) * L * circle' + mu)';

% main loop
for i = 1:np
    % correction; we might have measurements at the initial position
    if ~isempty(z{i,2})
        for j = 1:length(z{i,2})
            if any(seen_landmarks == z{i,2}(j)) % old landmark, add a loop-closure
                l_id = z{i,2}(j);
                % assemble the Jacobian for the jth measurement
                Hj = zeros(2, ekf.state_dim);
                Hj(:, 1:2) = H;
                Hj(:, 2+2*l_id-1:2+2*l_id) = J;
                % innovation covariance
                S = Hj * ekf.predict_cov * Hj' + Sigma_z;
                % Kalman gain
                K = ekf.predict_cov * Hj' * (S \ eye(2));
                % innovation
                v = z{i,1}(j,:)' - Hj * ekf.predict_state;
                % apply correction
                ekf.state = ekf.predict_state + K * v;
                ekf.cov = ekf.predict_cov - K * S * K';
            else % new landmark, initialize
                % append landmark id
                seen_landmarks = [seen_landmarks, z{i,2}(j)];
                l_id = z{i,2}(j);
                ekf.state(2+2*l_id-1:2+2*l_id) = ekf.state(1:2) + z{i,1}(j,:)';
                ekf.cov(2+2*l_id-1:2+2*l_id, 2+2*l_id-1:2+2*l_id) = ekf.cov(1:2, 1:2) + Sigma_z;
                ekf.cov(2+2*l_id-1:2+2*l_id, 1:2) = ekf.cov(1:2, 1:2);
                ekf.cov(1:2, 2+2*l_id-1:2+2*l_id) = ekf.cov(1:2, 1:2);
            end
        end
    end
    
    % prediction (propagation)
    % map
    ekf.predict_state(3:end) = ekf.state(3:end);
    % robot position
    if i < np
        ekf.predict_state(1:2,1) = ekf.state(1:2,1) + u(i,:)';
        ekf.predict_cov = ekf.cov + blkdiag(Sigma_u, zeros(2*size(map,1)));
    else
        ekf.predict_state(1:2,1) = ekf.state(1:2,1);
        ekf.predict_cov = ekf.cov;
    end

    % plotting
    for j = 1:size(z{i},1)
        line([x_gt(i,1), map_gt{i,1}(j,1)], [x_gt(i,2), map_gt{i,1}(j,2)],'Color',[green, 0.2],'LineStyle','-', 'linewidth', 1.5)
    end
    % plot odometry trajectory
    plot(x_odom(i,1), x_odom(i,2), '.', 'color', [VermillionRed, 0.7], 'MarkerSize', 18)
    if i > 1
        h_leg{2} = plot(x_odom(i-1:i,1), x_odom(i-1:i,2), ':', 'color', [VermillionRed, 0.7], 'linewidth', 2);
    end
    h_leg{3} = plot(ekf.state(1), ekf.state(2), '.', 'color', [darkblue, 0.7], 'MarkerSize', 22);
    if i > 1
        ellipse = (sqrt(Chi2) * chol(ekf.cov(1:2, 1:2)) * circle' + ekf.state(1:2))';
        plot(ellipse(:,1), ellipse(:,2), 'color', darkblue, 'linewidth', 1);
    else
        [~,p] = chol(ekf.init_cov(1:2, 1:2));
        if ~p
            ellipse = (sqrt(Chi2) * chol(ekf.init_cov(1:2, 1:2)) * circle' + ekf.init_state(1:2))';
            plot(ellipse(:,1), ellipse(:,2), 'color', darkblue, 'linewidth', 1);
        end
    end
    
    drawnow
    pause(0.1)
end

% SLAM map
h_leg{4} = plot(ekf.state(3:2:end), ekf.state(4:2:end), 'p', 'color', [darkblue, 0.99], 'MarkerSize', 14);
% draw 95% uncertainty ellipse
for i = 1:size(map,1)
    ellipse = (sqrt(Chi2) * chol(ekf.cov(2+2*i-1:2+2*i, 2+2*i-1:2+2*i)) * circle' + ekf.state(2+2*i-1:2+2*i))';
    plot(ellipse(:,1), ellipse(:,2), 'color', darkblue, 'linewidth', 1);
end
legend([h_leg{1}, h_leg{2}, h_leg{3}, h_leg{4}], 'Ground truth', 'Odometry', 'EKF-Robot', 'EKF-Map', 'location', 'best')

figure; hold on
figuresize(21,21,'cm')
title('Covariance Matrix $\Sigma$', 'fontsize',20, 'Interpreter','latex')
spy(abs((ekf.cov+ekf.cov')/2) > eps)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

figure; hold on
figuresize(21,21,'cm')
title('Information Matrix $\Sigma^{-1}$', 'fontsize',20, 'Interpreter','latex')
spy(abs((ekf.cov+ekf.cov')/2)\eye(ekf.state_dim) > eps)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
