% Toy example for tracking a single target using batch nonlinear least
% squares (Gauss-Newton).
%
%   Author: Maani Ghaffari Jadidi
%   Date:   03/06/2019

clc; clear; close all

% First simulate a target that moves on a curved path; we assume ownship is at
% the origin (0,0) and receives noisy range and bearing measurements of the
% target location. There is no knowledge of the target motion, but we
% assume target is close to its previous location to constrain the state.

% ground truth data
gt = [];
gt.x = -5:0.1:5;
gt.y = 1 * sin(gt.x) + 3;

% measurements
R = diag([0.1, 0.05].^2);
% Cholesky factor of covariance for sampling
Lz = chol(R, 'lower');
z = [];
for i = 1:length(gt.x)
   % sample from a zero mean Gaussian with covariance V
   noise = Lz * randn(2,1); 
   z(:,i) = [sqrt(gt.x(i)^2 + gt.y(i)^2); atan2(gt.x(i), gt.y(i))] + noise;
end

% Jacobians of the motion model ( we assume the transition model is
% identity, this will enforce the adjacent points remain close).
F = -eye(2);
G = eye(2);
Sigma_u = diag([0.05^2, 0.05^2]);
Lu = chol(Sigma_u, 'lower');

% Jacobian of measurement model
h = @(x)  [sqrt(x(1)^2 + x(2)^2); atan2(x(1), x(2))];
H = @(x) [x(1)/(x(1)^2 + x(2)^2)^(1/2), x(2)/(x(1)^2 + x(2)^2)^(1/2);
              x(2)/(x(1)^2 + x(2)^2), -x(1)/(x(1)^2 + x(2)^2)];
          
% initial guess; in general for a nonlinear problem finding a good initial
% guess can be difficult. Local solvers are sensitive to initial guess and
% only find local minimum. Here using each range and bearing measurement,
% the target position is fully observable so we can initialize using given
% noisy measurements. 
x_init = [];
for i = 1:size(z,2)
    x_init(2*i-1, 1) = z(1,i) * sin(z(2,i));
    x_init(2*i, 1) = z(1,i) * cos(z(2,i));
end

% constructing A using target motion 
nt = 2; % dimension of the target position
Sigma_init = diag([0.5^2, 0.5^2]);
A = chol(Sigma_init, 'lower') \ G; % Apply noise covariance as weight
for i = 2:size(z,2)
    A = blkdiag(A, Lu \ G);
    A(end-nt+1:end, end-2*nt+1:end-nt) = Lu \ F;
end

% Compute b (right hand side) for linear constraints
b = [];
for i = 1:size(z,2)
    b = [b; Lu \ zeros(2,1)];
end
nb = length(b);

% filling A using sensor measurements
for i = 1:size(z,2)
    [nA, mA] = size(A);
    Asub = zeros(nt, mA); % new submatrix in the bottom of A
    % fill in Jacobian of the corresponding target position
    Asub(:, 2*i-1:2*i) = Lz \ H(x_init(2*i-1:2*i));
    % append A and b
    A = [A; Asub];         % append A
end
A = sparse(A);

% Gauss-Newton solver
x_target = x_init;
max_iter = 1000;
iter = 0;
eps_Jr = 1e-6;
while iter < max_iter
    iter = iter + 1;
    % compute residual
    r = [];
    b = x_target(1:2);
    for i = 1:size(z,2)
        r(2*i-1:2*i,1) =  Lz \ (z(:,i) - h(x_target(2*i-1:2*i)));
        if i < size(z,2)
            b = [b; Lu \ (x_target(2*i+1:2*i+2) - x_target(2*i-1:2*i))];
        end
        % fill in Jacobian of the corresponding target position
        idx = nb + 2*i-1;
        A(idx:idx+1, 2*i-1:2*i) = Lz \ -H(x_target(2*i-1:2*i));
    end
    r = [b; r];
    % solve normal equations
    Jr = -A' * r;
    dx = (A' * A) \ Jr;
    x_target = x_target + dx;
    % check if converged
    if norm(Jr) < eps_Jr
        break;
    end
end

% plotting
green = [0.2980 .6 0];
darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
VermillionRed = [156,31,46]/255;

fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
alpha = 0.8;

figure; hold on
plot(0,0, '^', 'MarkerFaceColor', 'b', 'markersize', 20)
plot(x_init(1:2:end), x_init(2:2:end), '-', 'color', [green, alpha], 'linewidth', 2.5)
plot(gt.x, gt.y, '-', 'color', [Darkgrey, alpha], 'linewidth', 2.5)
plot(x_target(1:2:end), x_target(2:2:end), '-', 'color', [VermillionRed, alpha], 'linewidth', 2.5)
legend('ownship', 'initial guess', 'ground truth', 'Gauss-Newton', 'location', 'best')
grid on, axis auto equal tight
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
figuresize(21,21,'cm')
% print -opengl -dpng -r600 nls_example_tracking.png