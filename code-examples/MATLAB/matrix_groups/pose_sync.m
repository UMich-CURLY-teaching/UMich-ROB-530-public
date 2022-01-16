% An example of pose, SE(3), syncronization. We bulild a pose graph where
% all factors are relative SE(3) measurements. Then we use Gauss-Newton to
% find a locally optimal solution.

%   Author: Maani Ghaffari Jadidi
%   Date:   03/09/2019

clc; clear; close all

fsize = 18; % font size
alpha = 0.2;
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
% colors
green = [0.2980 .6 0];
crimson = [220,20,60]/255; 
darkblue = [0 .2 .4];
DupontGray = [144,131,118]/255;

% se(3) generators; twist = vec(omega, v).
% R^3 standard basis
e1 = [1;0;0];
e2 = [0;1;0];
e3 = [0;0;1];
% se(3) (SE(3) Lie algebra) infinitesimal generators
G = cell(6,1);
G{1} = [skew(e1), zeros(3,1); zeros(1,4)];
G{2} = [skew(e2), zeros(3,1); zeros(1,4)];
G{3} = [skew(e3), zeros(3,1); zeros(1,4)];
G{4} = zeros(4); G{1}(1,4) = 1;
G{5} = zeros(4); G{2}(2,4) = 1;
G{6} = zeros(4); G{3}(3,4) = 1;

% a sequence of relative rigid body transformation
U = cell(5,1);
U{1} = [Rzyx(0.1, 0.01, -0.02), [0.8; 0.6; 0.9]; 0 0 0 1];      % T01
U{2} = [Rzyx(0.1, -0.1, 0.2), [0.9; -0.6; 0.6]; 0 0 0 1];       % T12
U{3} = [Rzyx(0.3, -0.1, -0.3), [1.2; 0.6; 1.0]; 0 0 0 1];       % T23
U{4} = [Rzyx(-0.3, -0.1, -0.1), [1.2; 0.9; 0.8]; 0 0 0 1];      % T34
U{5} = [Rzyx(-0.7, -0.1, -0.3), [-1.4; 1.5; -1.1]; 0 0 0 1];    % T45

% accumulate the transformations, strating from the identity
Keys = []; % to track nodes in each factors 
T = cell(6,1);
T{1} = eye(4);
for i = 2:length(T)
    T{i} = T{i-1} * U{i-1};
    Keys{i-1} = [i-1, i];
end

% create noisy SE(3) measurements
% measurement noise covariance
Sigma = diag([0.05^2, 0.03^2, 0.03^2, 0.05^2, 0.05^2, 0.05^2]);
% Cholesky factor of covariance for sampling
Lz = chol(Sigma, 'lower');
Z = cell(5,1);
for i = 1:length(Z)
    % sample from a zero mean Gaussian
    noise = Lz * randn(6,1);
    N = zeros(4);
    for j = 1:length(G)
        N = N + G{j} * noise(j);
    end
    Z{i} = U{i} * expm(N);
end

% add two loop-closures between T1-T5, and T3-T5
l1 = [1, 4];
l2 = [6, 6];
nz = length(Z);
for i = 1:length(l1)
    % sample from a zero mean Gaussian
    noise = Lz * randn(6,1);
    N = zeros(4);
    for j = 1:length(G)
        N = N + G{j} * noise(j);
    end
    Z{nz+i} = (T{l1(i)} \ T{l2(i)}) * expm(N);
    Keys{nz+i} = [l1(i), l2(i)];
end

% plot the pose graph
figure;
hold on; grid on; axis auto 
text_offset = [-0.1; -0.1; 0.5];
h_leg = [];
for i = 1:length(T)
    h_leg{1} = plot3(T{i}(1,4), T{i}(2,4), T{i}(3,4), 'o', 'color', ...
        [crimson, alpha], 'MarkerFaceColor', crimson, 'markersize', 10);
    text(T{i}(1,4) + text_offset(1), T{i}(2,4) + text_offset(2), T{i}(3,4) + text_offset(3), ...
        ['T_',num2str(i-1)], 'FontSize', fsize)
    if i > 1
        x = [T{i-1}(1,4), T{i}(1,4)];
        y = [T{i-1}(2,4), T{i}(2,4)];
        z = [T{i-1}(3,4), T{i}(3,4)];
        line(x, y, z, 'Color', [green, alpha], 'LineStyle', '-', 'linewidth', 4)
    end
end
% plot loop-closures
k = 1;
for i = length(T):length(Z)
    x = [T{l1(k)}(1,4), T{l2(k)}(1,4)];
    y = [T{l1(k)}(2,4), T{l2(k)}(2,4)];
    z = [T{l1(k)}(3,4), T{l2(k)}(3,4)];
    line(x, y, z, 'Color', [green, alpha], 'LineStyle', '-', 'linewidth', 4)
    k = k + 1;
end


% compute initial guess by accumulating noisy measurements
T_init = cell(6,1);
T_init{1} = eye(4);
for i = 2:length(T_init)
    T_init{i} = T_init{i-1} * Z{i-1};
end

% plot initial guess 
for i = 1:length(T_init)
    h_leg{2} = plot3(T_init{i}(1,4), T_init{i}(2,4), T_init{i}(3,4), 's', 'color', ...
        [DupontGray, alpha], 'MarkerFaceColor', DupontGray, 'markersize', 10);
%     text(T_init{i}(1,4) - text_offset(1)/2, T_init{i}(2,4) - text_offset(2)/2, T_init{i}(3,4) - text_offset(3)/2, ...
%         ['T^{init}_',num2str(i-1)], 'FontSize', fsize)
    if i > 1
        x = [T_init{i-1}(1,4), T_init{i}(1,4)];
        y = [T_init{i-1}(2,4), T_init{i}(2,4)];
        z = [T_init{i-1}(3,4), T_init{i}(3,4)];
        line(x, y, z, 'Color', [DupontGray, alpha], 'LineStyle', '-', 'linewidth', 4)
    end
end

% We now construct the Jacobian matrix. The rows are the measurements which are
% SE(3) here. Therefore, each measurement occupies 6 rows. Columns
% correspond to the decision variables which are SE(3) here. Therefore, we
% have 6 x number of poses variables. Note that the optimization is
% parametrized uing twist (6x1) which lives in the Lie algebra se(3). We
% find the corretion twist and retract it using Lie exp map and "add"
% (multiply) it to the previous iteration guess. This process should be
% repeated for each pose before moving to the next iteration. Further, we
% anchor the first pose to constrain the graph, i.e., we fix the first pose
% to be at the identity. This will all an extra 6 rows to the jacobian
% matrix and is equivalent of placing a prior over the first node of the
% graph.

% Jacobian matrix
A = zeros(6 + 6 * length(Z), 6 * length(T_init));
% right hand side (residuals)
b = zeros(6 + 6 * length(Z),1);
% anchor node covariance; we want to fix the node so the covariance should
% be small. This will result in large weights in the optimization process.
Sigma_init = eye(6) * 0.1^2;
A(1:6,1:6) = chol(Sigma_init, 'lower') \ eye(6);

% Gauss-Newton solver over SE(3)
T_est = T_init;
max_iter = 100;
iter = 0;
eps_Jr = 1e-9;
dx = zeros(6 + 6 * length(Z),1);
% SE(3) right Jacobian inverse and adjoint
invJr = @RightJacobianInverse_SE3;
Ad = @Adjoint_SE3;
while iter < max_iter
    iter = iter + 1;
    % compute residual
    r = dx;
    r(1:6,1) = wedge(logm(T_est{1}));
%     b = x_target(1:2);
    for i = 1:length(Keys)
        key = Keys{i};
        res_idx = 6*i+1:6*i+6;
        r(res_idx, 1) =  wedge(logm(Z{i} \ (T_est{key(1)} \  T_est{key(2)})));
        % fill in Jacobian of the corresponding target position
        idx = 6 * key(1) - 5;
        A(res_idx, idx:idx+5) = Lz \ -invJr(r(res_idx, 1)) * Ad(T_est{key(2)} \  T_est{key(1)});
        idx = 6 * key(2) - 5;
        A(res_idx, idx:idx+5) = Lz \ invJr(r(res_idx, 1));
        r(res_idx, 1) =  Lz \ r(res_idx, 1);
    end
    % solve normal equations
    Jr = -A' * r;
    dx = (A' * A) \ Jr;
    disp(iter)
    % retract and update the estimate
    for i = 1:length(T_est)
        xi = dx(6*i-5:6*i);
        T_est{i} = T_est{i} * expm(hat(xi));
    end
    % check if converged
    if norm(Jr) < eps_Jr
        break;
    end
end

% plot result
for i = 1:length(T_est)
    h_leg{3} = plot3(T_est{i}(1,4), T_est{i}(2,4), T_est{i}(3,4), 's', 'color', ...
        [darkblue, alpha], 'MarkerFaceColor', darkblue, 'markersize', 12);
%     text(T_est{i}(1,4) - text_offset(1)/2, T_est{i}(2,4) - text_offset(2)/2, T_est{i}(3,4) - text_offset(3)/2, ...
%         ['T^{GN}_',num2str(i-1)], 'FontSize', fsize)
    if i > 1
        x = [T_est{i-1}(1,4), T_est{i}(1,4)];
        y = [T_est{i-1}(2,4), T_est{i}(2,4)];
        z = [T_est{i-1}(3,4), T_est{i}(3,4)];
        line(x, y, z, 'Color', [darkblue, alpha], 'LineStyle', '-', 'linewidth', 4)
    end
end

legend([h_leg{1}, h_leg{2}, h_leg{3}], 'ground truth', 'initial guess', 'Gauss-Newton', ...
    'location', 'best', 'fontsize', fsize)
axis equal tight
view(-150,25)
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
% print -opengl -dpng -r600 pose_sync_se3.png


%% define SE(3) methods
function X = skew(x)
% vector to skew R^3 -> so(3)
X = [   0,  -x(3),  x(2);
    x(3),      0,  -x(1);
    -x(2), x(1),   0];
end

function x = unskew(X)
% so(3) -> R^3
x = [X(3,2); X(1,3); X(2,1)];
end

function X = hat(x)
% hat: R^6 -> se(3)
X = [skew(x(1:3)), x(4:6); 0 0 0 0];
end

function x = wedge(X)
% wedge: se(3) -> R^6
x = [unskew(X(1:3,1:3)); X(1:3,4)];
end

function Ad = Adjoint_SE3(X)
% ADJOINT_SE3 Computes the adjoint of SE(3)
Ad = [X(1:3,1:3), zeros(3); ...
      skew(X(1:3,4))*X(1:3,1:3), X(1:3,1:3)];
end

function output = LeftJacobian_SO3(w)
%LEFT JACOBIAN as defined in http://perpustakaan.unitomo.ac.id/repository/Associating%20Uncertainty%20With%20Three-Dimensional.pdf
theta = norm(w);
A = skew(w);
if theta == 0
    output = eye(3);
    return;
end
output = eye(3) + ((1-cos(theta))/theta^2)*A + ((theta-sin(theta))/theta^3)*A^2;
end

function output = LeftJacobian_SE3(xi)
% https://github.com/RossHartley/lie/blob/master/matlab/%2BLie/LeftJacobian_SE3.m
% RIGHT JACOBIAN as defined in http://perpustakaan.unitomo.ac.id/repository/Associating%20Uncertainty%20With%20Three-Dimensional.pdf

Phi = xi(1:3);
phi = norm(Phi);
Rho = xi(4:6);
Phi_skew = skew(Phi);
Rho_skew = skew(Rho);
J = LeftJacobian_SO3(Phi);

if (phi == 0)
    Q = 0.5*Rho_skew;
else
    Q = 0.5*Rho_skew ...
        + (phi-sin(phi))/phi^3 * (Phi_skew*Rho_skew + Rho_skew*Phi_skew + Phi_skew*Rho_skew*Phi_skew) ...
        - (1-0.5*phi^2-cos(phi))/phi^4 * (Phi_skew*Phi_skew*Rho_skew + Rho_skew*Phi_skew*Phi_skew - 3*Phi_skew*Rho_skew*Phi_skew) ...
        - 0.5*((1-0.5*phi^2-cos(phi))/phi^4 - 3*(phi-sin(phi)-(phi^3)/6)/phi^5) ...
        * (Phi_skew*Rho_skew*Phi_skew*Phi_skew + Phi_skew*Phi_skew*Rho_skew*Phi_skew);
end
output = [J, zeros(3); Q, J];
end

function output = RightJacobian_SE3(xi)
output = Adjoint_SE3(expm(hat(-xi))) * LeftJacobian_SE3(xi);
end

function output = RightJacobianInverse_SE3(xi)
Jr = RightJacobian_SE3(xi);
output = Jr \ eye(size(Jr));
end


