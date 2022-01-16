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
hold on; grid on; axis auto
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
for i = 1:size(x_gt,1)
    Idx = rangesearch(MdlKDT, x_gt(i,:), z_max);
    map_gt = map(Idx{1},:);
    z{i,1} = map_gt - x_gt(i,:) + (Lz * randn(size(map_gt,1),2)')'; % landmark coordinates
    z{i,2} = Idx{1};        % correspondences
    for j = 1:size(z{i},1)
        line([x_gt(i,1), map_gt(j,1)], [x_gt(i,2), map_gt(j,2)],'Color',[green, 0.2],'LineStyle','-', 'linewidth', 1.5)
    end
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

% plot odometry trajectory
h_leg{2} = plot(x_odom(:,1), x_odom(:,2), ':', 'color', [VermillionRed, 0.7], 'linewidth', 2);
plot(x_odom(:,1), x_odom(:,2), '.', 'color', [VermillionRed, 0.7], 'MarkerSize', 18)


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

% constructing A using robot motion 
A = Sigma_init \ G; % Full Jacobian
for i = 1:size(x_odom,1)-1
    A = blkdiag(A, Lu \ G);
    A(end-nr+1:end, end-2*nr+1:end-nr) = Lu \ F;
end
% measurements must be filled based on correspondences
% for new landmarks A has to be expanded whereas for old ones the
% correspondences must be found.
seen_landmarks = [];    % list of seen landmarks

% Compute b (right hand side)
b = Lu \ x_init';
for i = 1:np-1
    b = [b; Lu \ u(i,:)'];
end

for i = 1:np
    if ~isempty(z{i,2})
        for j = 1:length(z{i,2})
            if any(seen_landmarks == z{i,2}(j)) % old landmark, add a loop-closure
                [nA, mA] = size(A);
                Asub = zeros(nl, mA); % new submatrix in the bottom of A
                % fill in Jacobian of the corresponding robot position/pose
                % and landmark
                Asub(:, nr*i-1:nr*i) = Lz \ H;
                % find the landmark id stored in A
                l_id = find(seen_landmarks == z{i,2}(j));
                A_id = nr * np + nl * (l_id-1);
                Asub(:, A_id+1:A_id+nl) = Lz \ J;
                % append A and b
                A = [A; Asub];          % append A
                b = [b; Lz \ z{i,1}(j,:)'];  % append b
            else % new landmark, exploration; expand the state and Jacobian
                % append landmark id
                seen_landmarks = [seen_landmarks, z{i,2}(j)];
                A = blkdiag(A, Lz \ J); % expand A based on J
                % Jacobian of the corresponding robot position/pose
                A(end-size(H,1)+1:end, nr*i-1:nr*i) = Lz \ H;
                b = [b; Lz \ z{i,1}(j,:)'];  % append b
            end
        end
    end
end

% Solve Ax=b for the robot trajectory and map
x_slam = A\b;

% Solve using QR factorization
[Q,R] = qr(A);
x_qr = R\Q'*b;

% Solve using Cholesky factorization
L = chol(A'*A, 'lower');
x_chol = L' \ (L \ (A'*b));

disp(['norm( A \ b - R \ Q^T * b) = ', num2str(norm(x_slam-x_qr))])
disp(['norm( A \ b - L^T \ (L \ A^T * b)) = ', num2str(norm(x_slam-x_chol))])


%% Plot results
% SLAM trajectory
h_leg{3} = plot(x_slam(1:2:nr*np), x_slam(2:2:nr*np), '-', 'color', [darkblue, 0.7], 'linewidth', 2);
plot(x_slam(1:2:nr*np), x_slam(2:2:nr*np), '.', 'color', [darkblue, 0.7], 'MarkerSize', 18);
% SLAM map
plot(x_slam(nr*np+1:2:end), x_slam(nr*np+2:2:end), 'p', 'color', [darkblue, 0.99], 'MarkerSize', 14)
legend([h_leg{1}, h_leg{2}, h_leg{3}], 'Ground truth', 'Odometry', 'SLAM', 'location', 'best')
axis equal
figuresize(21,21,'cm')

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
