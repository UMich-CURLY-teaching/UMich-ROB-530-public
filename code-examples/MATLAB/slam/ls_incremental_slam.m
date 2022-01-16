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

clc; clear; close all

fsize = 16; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

% nice colours
green = [0.2980 .6 0];
crimson = [220,20,60]/255; 
darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
darkgrey = [.35 .35 .35];
lightgrey = [.7 .7 .7];
Lightgrey = [.9 .9 .9];
VermillionRed = [156,31,46]/255;
DupontGray = [144,131,118]/255;
Azure = [53, 112, 188]/255;
purple = [178, 102, 255]/255;
orange = [255,110,0]/255;

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
z_max = 1.5;      % maximum sensor range in meters
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
nr = 2; % dimension of the robot position/pose
nl = 2; % dimension of landmark
np = size(x_odom,1); % number of poses
% Jacobians of the motion model
F = -eye(2);
G = eye(2);
% Jacobian of measurement model
H = -eye(2);     % wrt to robot position
J = eye(2);      % wrt to landmark position

% measurements must be filled based on correspondences
% for new landmarks A has to be expanded whereas for old ones the
% correspondences must be found.
seen_landmarks = [];    % list of seen landmarks
landmark_key = [];
for i = 1:np
    if i == 1
        % initialize b (right hand side)
        b = Lu \ x_init';
        % initialize A using initial condition
        A = Sigma_init \ G;
        pose_key = 1;
        % initialize Cholesky factor R = L^T
        R = chol(A'*A);
    else
        % Compute b (right hand side)
        b = [b; Lu \ u(i-1,:)'];
        % constructing A using robot motion 
        A = blkdiag(A, Lu \ G);
        A(end-nr+1:end, pose_key(i-1):pose_key(i-1)+1) = Lu \ F;
        % update pose key
        pose_key = [pose_key; size(A,2)-nr+1];
        % expand Cholesky factor
        R = blkdiag(R, zeros(size(J)));
        for k = nl:-1:1
            x = A(end-k+1, :)';
            R = cholupdate(R,x);
        end
    end
    
    if ~isempty(z{i,2})
        for j = 1:length(z{i,2})
            if any(seen_landmarks == z{i,2}(j)) % old landmark, add a loop-closure
                [nA, mA] = size(A);
                Asub = zeros(nl, mA); % new submatrix in the bottom of A
                % find the landmark id stored in A
                A_id = landmark_key(seen_landmarks == z{i,2}(j));
                % add Jacobian of the corresponding landmark and robot position/pose
                Asub(:, A_id:A_id+nl-1) = Lz \ J;
                Asub(:, pose_key(i):pose_key(i)+1) = Lz \ H;
                % append A and b                
                A = [A; Asub];
                b = [b; Lz \ z{i,1}(j,:)'];  % append b
                % expand Cholesky factor
                for k = nr:-1:1
                    x = A(end-k+1, :)';
                    R = cholupdate(R,x);
                end
            else % new landmark, exploration; expand the state and Jacobian
                % append landmark id
                seen_landmarks = [seen_landmarks, z{i,2}(j)];
                A = blkdiag(A, Lz \ J); % expand A based on J
                % Jacobian of the corresponding robot position/pose
                A(end-size(H,1)+1:end, pose_key(i):pose_key(i)+1) = Lz \ H;
                b = [b; Lz \ z{i,1}(j,:)'];  % append b
                % update landmark key
                landmark_key = [landmark_key; size(A,2)-1];
                % expand Cholesky factor
                R = blkdiag(R, zeros(size(J)));
                for k = nl:-1:1
                    x = A(end-k+1, :)';
                    R = cholupdate(R,x);
                end
            end
        end
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
    
    drawnow
    pause(0.1)
end

x_slam = R \ (R' \ (A'*b));
% Final SLAM trajectory
h_leg{3} = plot(x_slam(pose_key), x_slam(pose_key+1), '-', 'color', [darkblue, 0.7], 'linewidth', 2);
plot(x_slam(pose_key), x_slam(pose_key+1), '.', 'color', [darkblue, 0.7], 'MarkerSize', 18);
% SLAM map
plot(x_slam(landmark_key), x_slam(landmark_key+1), 'p', 'color', [darkblue, 0.99], 'MarkerSize', 14)
legend([h_leg{1}, h_leg{2}, h_leg{3}], 'Ground truth', 'Odometry', 'SLAM', 'location', 'best')


% Solve using QR factorization
[Q,R1] = qr(A);
x_qr = R1\Q'*b;

% Solve using Cholesky factorization
L = chol(A'*A, 'lower');
x_chol = L' \ (L \ (A'*b));

disp(['norm( A \ b - R \ Q^T * b) = ', num2str(norm(x_slam-x_qr))])
disp(['norm( A \ b - L^T \ (L \ A^T * b)) = ', num2str(norm(x_slam-x_chol))])


% print -painters -dpdf -r600 ls_slam.pdf
% print -opengl -dpng -r600 ls_slam.png

figure;
figuresize(21,21,'cm')

subplot(2,2,1); hold on
title('$\textbf{A}$', 'fontsize',30, 'Interpreter','latex')
spy(A)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,2); hold on
title('$\textbf{A}^\mathsf{T} \textbf{A}$', 'fontsize',30, 'Interpreter','latex')
spy(A'*A)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,3); hold on
title('$\textbf{R}$ - QR', 'fontsize',30, 'Interpreter','latex')
spy(R(1:size(A,2),:))
h = get(gca,'children');
h.Color = darkblue;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,4); hold on
title('$\textbf{R}$ - Cholesky', 'fontsize',30, 'Interpreter','latex')
spy(L') % transpose to get upper triangular
h = get(gca,'children');
h.Color = VermillionRed;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

% print -painters -dpdf -r600 ls_slam_sparsity.pdf
% print -opengl -dpng -r600 ls_slam_sparsity.png

%% Variable reordering to improve sparsity; COLAMD: COLumn Approximate Minimum Degree permutation
% A
p = colamd(A);
Anew = A(:,p);
% compute new R
[~, Rnew] = qr(Anew);
% compute new L
Lnew = chol(Anew' * Anew, 'lower'); % Upper triangular by default, use 'lower' for lower triangular

figure;
figuresize(21,21,'cm')

subplot(2,2,1); hold on
title('$\textbf{A}$ + COLAMD', 'fontsize',30, 'Interpreter','latex')
spy(Anew)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,2); hold on
title('$\textbf{A}^\mathsf{T} \textbf{A}$', 'fontsize',30, 'Interpreter','latex')
spy(Anew' * Anew)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,3); hold on
title('$\textbf{R}$ - QR', 'fontsize',30, 'Interpreter','latex')
spy(Rnew(1:size(A,2),:))
h = get(gca,'children');
h.Color = darkblue;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,4); hold on
title('$\textbf{R}$ - Cholesky', 'fontsize',30, 'Interpreter','latex')
spy(Lnew')
h = get(gca,'children');
h.Color = VermillionRed;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

% print -painters -dpdf -r600 ls_slam_sparsity_colamd.pdf
% print -opengl -dpng -r600 ls_slam_sparsity_colamd.png

%% Variable reordering to improve sparsity; COLAMD: COLumn Approximate Minimum Degree permutation
% A
p = colperm(A);
Anew = A(:,p);
% compute new R
[~, Rnew] = qr(Anew);
% compute new L
Lnew = chol(Anew' * Anew); % Upper triangular by default, use 'lower' for lower triangular

figure;
figuresize(21,21,'cm')

subplot(2,2,1); hold on
title('$\textbf{A}$ + COLPERM', 'fontsize',30, 'Interpreter','latex')
spy(Anew)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,2); hold on
title('$\textbf{A}^\mathsf{T} \textbf{A}$', 'fontsize',30, 'Interpreter','latex')
spy(Anew' * Anew)
h = get(gca,'children');
h.Color = Darkgrey;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,3); hold on
title('$\textbf{R}$ - QR', 'fontsize',30, 'Interpreter','latex')
spy(Rnew(1:size(A,2),:))
h = get(gca,'children');
h.Color = darkblue;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,4); hold on
title('$\textbf{R}$ - Cholesky', 'fontsize',30, 'Interpreter','latex')
spy(Lnew)
h = get(gca,'children');
h.Color = VermillionRed;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

% print -painters -dpdf -r600 ls_slam_sparsity_colperm.pdf
% print -opengl -dpng -r600 ls_slam_sparsity_colperm.png
