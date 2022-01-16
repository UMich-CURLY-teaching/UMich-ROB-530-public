% Toy example for tracking a single target using batch least squares
%
%   Author: Maani Ghaffari Jadidi
%   Date:   02/28/2019

clc; clear; close all

% First simulate a target that moves on a curved path; we assume ownship is at
% the origin (0,0) and receives direct but noisy (x,y) measurements of the
% target location. There is no knowledge of the target motion, but we
% assume target is close to its previous location to constrain the state.

% ground truth data
gt = [];
gt.x = -5:0.1:5;
gt.y = 1 * sin(gt.x) + 3;

% measurements
R = diag([0.05, 0.05].^2);
% Cholesky factor of covariance for sampling
Lz = chol(R, 'lower');
z = [];
for i = 1:length(gt.x)
   % sample from a zero mean Gaussian with covariance V
   noise = Lz * randn(2,1); 
   z(:,i) = [gt.x(i); gt.y(i)] + noise;
end

% Jacobians of the motion model ( we assume the transition model is
% identity, this will enforce the adjacent points remain close).
F = -eye(2);
G = eye(2);
Sigma_u = diag([0.03^2, 0.03^2]);
Lu = chol(Sigma_u, 'lower');

% Jacobian of measurement model
H = eye(2);

% constructing A using target motion 
nt = 2; % dimension of the target position
Sigma_init = diag([0.8^2, 0.8^2]);
A = chol(Sigma_init, 'lower') \ G; % Apply noise covariance as weight
for i = 2:size(z,2)
    A = blkdiag(A, Lu \ G);
    A(end-nt+1:end, end-2*nt+1:end-nt) = Lu \ F;
end

% Compute b (right hand side)
b = [];
for i = 1:size(z,2)
    b = [b; Lu \ zeros(2,1)];
end

% filling A using sensor measurements
for i = 1:size(z,2)
    [nA, mA] = size(A);
    Asub = zeros(nt, mA); % new submatrix in the bottom of A
    % fill in Jacobian of the corresponding target position
    Asub(:, 2*i-1:2*i) = Lz \ H;
    % append A and b
    A = [A; Asub];         % append A
    b = [b; Lz \ z(:,i)];  % append b
end
A = sparse(A);

% Solve Ax=b for the target trajectory
x_target = A\b;

% Solve using QR factorization
[Q,R] = qr(A);
x_qr = R\Q'*b;

% Solve using Cholesky factorization
L = chol(A'*A, 'lower');
x_chol = L' \ (L \ (A'*b));

disp(['norm( A \ b - R \ Q^T * b) = ', num2str(norm(x_target-x_qr))])
disp(['norm( A \ b - L^T \ (L \ A^T * b)) = ', num2str(norm(x_target-x_chol))])

% plotting
darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
VermillionRed = [156,31,46]/255;

fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

figure; hold on
plot(0,0, '^', 'MarkerFaceColor', 'b', 'markersize', 20)
plot(gt.x, gt.y, '-', 'linewidth', 2)
plot(x_target(1:2:end), x_target(2:2:end), '-k', 'linewidth', 2)
legend('ownship', 'ground truth', 'Least Squares', 'location', 'best')
grid on, axis auto equal tight
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
figuresize(21,21,'cm')
% print -opengl -dpng -r600 ls_example_tracking.png

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
title('$\mathbf{R}$ - QR', 'fontsize',30, 'Interpreter','latex')
spy(R(1:size(A,2),:))
h = get(gca,'children');
h.Color = darkblue;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,4); hold on
title('$\mathbf{R}$ - Cholesky', 'fontsize',30, 'Interpreter','latex')
spy(L') % transpose to get upper triangular
h = get(gca,'children');
h.Color = VermillionRed;
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
