% An example of process models on Lie groups and uncertainty propagation
% using SE(3). The process model is simply X_k+1 = X_k * exp(U_k + w_k) where
% X_k is in SE(3) and U_k and w_k (zero mean white Gaussian noise, N(0,Q_k)) 
% are defined in the Lie algebra se(3). We use Monte Carlo methods to propagate 
% samples over a path and then compute the sample mean and covariance on Lie group, 
% here SE(3). Note that the sample mean and covariance are computed using an
% iterative algorithm which is different than usual Euclidean sample
% statistics. The covariance on Lie algebra is flat as expected but it's
% nonlinear when mapped to the manifold using Lie exp map.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   03/05/2019

clc; clear; close all

% se(3) generators; twist = vec(v, omega).
% R^3 standard basis
e1 = [1;0;0];
e2 = [0;1;0];
e3 = [0;0;1];
% so(3) (SO(3) Lie algebra) infinitesimal generators
robot = [];
robot.G1 = zeros(4);
robot.G1(1,4) = 1;

robot.G2 = zeros(4);
robot.G2(2,4) = 1;

robot.G3 = zeros(4);
robot.G3(3,4) = 1;

robot.G4 = [skew(e1), zeros(3,1)
    zeros(1,4)];

robot.G5 = [skew(e2), zeros(3,1)
    zeros(1,4)];

robot.G6 = [skew(e3), zeros(3,1)
    zeros(1,4)];

% SE(3) Adjoint
robot.Ad = @(X) [X(1:3,1:3), skew(X(1:3,4))*X(1:3,1:3); zeros(3), X(1:3,1:3)];

% generate noise-free control inputs in Lie algebra se(3)
nstep = 4;
u = [];
u(1,:) = linspace(0, 4, nstep);
u(2,:) = 1.5 * u(1,:);
u(3,:) = 0.05 * exp(0.6 * u(1,:)) - 0.05;
u(4,:) = linspace(0, 0.5, nstep);
u(5,:) = linspace(0, 0.3, nstep);
u(6,:) = linspace(0, 0.7, nstep);

% build a 3D robot!
robot.n = 1000;
robot.x = cell(robot.n,1); % state mean
robot.Cov_fo = zeros(6); % first order covariance propagation around mean
for i = 1:robot.n
    robot.x{i} = eye(4);
end
% motion model noise covariance
robot.Q = diag([0.01^2, 0.01^2, 0.05^2, 0.05^2, 0.05^2, 0.2^2]);
% Cholesky factor of covariance for sampling
robot.L = chol(robot.Q, 'lower');

% construct noise free motion trajectory
path = [];
path.T = eye(4);
path.x = 0;
path.y = 0;
path.z = 0;
for i = 1:size(u,2)
    Ui = u(1,i) * robot.G1 + u(2,i) * robot.G2 + u(3,i) * robot.G3 + ...
        u(4,i) * robot.G4 + u(5,i) * robot.G5 + u(6,i) * robot.G6;
    path.T = path.T * expm(Ui);
    path.x(i+1) = path.T(1,4);
    path.y(i+1) = path.T(2,4);
    path.z(i+1) = path.T(3,4);
end

% create confidence ellipsoid
% first create points from a unit sphere
phi = linspace(-pi, pi, 100)';
theta = linspace(-pi/2, pi/2, 100)';
[PHI, THETA] = meshgrid(phi, theta);
X_sph = cos(THETA) .* cos(PHI);
Y_sph = cos(THETA) .* sin(PHI);
Z_sph = sin(THETA);
sphere = [X_sph(:), Y_sph(:), Z_sph(:)];
sphere = [sphere, zeros(size(sphere))];
% Chi-squared 6-DOF 95% percent confidence (0.05): 12.592	
scale = sqrt(12.592);

% incremental visualization
green = [0.2980 .6 0];
crimson = [220,20,60]/255;
darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
VermillionRed = [156,31,46]/255;
DupontGray = [144,131,118]/255;

fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

h_leg = []; % legend handle
figure; hold on
h_leg{1} = plot3(path.x, path.y, path.z, '-', 'color', Darkgrey, 'linewidth', 3);
grid on, axis auto equal%, axis([-6 6 -1 5])
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
zlabel('$x_3$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
axis equal, view(69,40)

% extract propagated particles
p = [];
for i = 1:robot.n
    p(1,i) = robot.x{i}(1,4);
    p(2,i) = robot.x{i}(2,4);
    p(3,i) = robot.x{i}(3,4);
end

% compute sample statistics
[mu, Sigma] = Lie_sample_statistics(robot);

% plot sample mean and particles
h_leg{2} = plot3(p(1,:), p(2,:), p(3,:), '.','Color', [green, .25], 'markersize', 14);
h_leg{3} = plot3(mu(1,4), mu(2,4), mu(3,4), 'o', 'color', crimson, 'MarkerFaceColor', crimson, 'markersize', 8);

% main loop; iterate over the control inputs and move the robot
ELLIPSOID = zeros(size(sphere,1),3); % covariance ellipse on manifold (nonlinear)
ELLIPSOID_fo = zeros(size(sphere,1),3); % first order covariance ellipse on manifold (nonlinear)
ellipsoid = zeros(size(sphere,1),3); % covariance ellipse on Lie algebra

for i = 1:size(u,2)
    % move particles based on the input
    robot = propagation(robot, u(:,i));
    % extract propagated particles
    p = [];
    for j = 1:robot.n
        p(1,j) = robot.x{j}(1,4);
        p(2,j) = robot.x{j}(2,4);
        p(3,j) = robot.x{j}(3,4);
    end
    
    % show particles
    set(h_leg{2},'XData', p(1,:),'YData', p(2,:),'ZData', p(3,:));
    
    % compute sample statistics
    [mu, Sigma] = Lie_sample_statistics(robot);
    
    % compute first order analytical covariance propagation
    Ui = expm(u(1,i) * robot.G1 + u(2,i) * robot.G2 + u(3,i) * robot.G3 + ...
        u(4,i) * robot.G4 + u(5,i) * robot.G5 + u(6,i) * robot.G6);
    
    % left-invariant error: eta^L = X^-1 * X^hat
    % robot.Ad(inv(Ui) maps the covariance back to Lie algebra using the
    % incremental motion Ui (hence inv(Ui). Then the noise covariance that
    % is already defined in Lie algebra can be added to the mapped state
    % covariance.
    robot.Cov_fo = robot.Ad(inv(Ui)) * robot.Cov_fo * robot.Ad(inv(Ui))' + robot.Q;
    
    robot.L_fo = chol(robot.Cov_fo, 'lower');
    
    % create the ellipse using the unit circle
    [L, flag] = chol(Sigma, 'lower');
    
    % if Sigma is positive definite plot the ellipse
    if ~flag
        for j = 1:size(sphere,1)
            % sample covariance on SE(2)
            ell_se3_vec = scale * L * sphere(j,:)';
            % retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
            temp = mu * expm(robot.G1 * ell_se3_vec(1) + robot.G2 * ell_se3_vec(2) + robot.G3 * ell_se3_vec(3) + ...
                robot.G4 * ell_se3_vec(4) + robot.G5 * ell_se3_vec(5) + robot.G6 * ell_se3_vec(6));
            ELLIPSOID(j,:) = [temp(1,4), temp(2,4), temp(3,4)];
            
            % get the ellipse on Lie algebra
            temp = mu * [Rzyx(ell_se3_vec(4), ell_se3_vec(5), ell_se3_vec(6)), ell_se3_vec(1:3); 
                        0 0 0 1];
            ellipsoid(j,:) = [temp(1,4), temp(2,4), temp(3,4)];
            
            % sample covariance on SE(3)
            ell_se3_vec = scale * robot.L_fo * sphere(j,:)';
            % retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
            temp = mu * expm(robot.G1 * ell_se3_vec(1) + robot.G2 * ell_se3_vec(2) + robot.G3 * ell_se3_vec(3) + ...
                robot.G4 * ell_se3_vec(4) + robot.G5 * ell_se3_vec(5) + robot.G6 * ell_se3_vec(6));
            ELLIPSOID_fo(j,:) = [temp(1,4), temp(2,4), temp(3,4)];
        end
        
        % plot the ellipsoids
        % extract x,y,z matrices for plotting
        X_ell = reshape(ELLIPSOID(:,1), size(X_sph));
        Y_ell = reshape(ELLIPSOID(:,2), size(Y_sph));
        Z_ell = reshape(ELLIPSOID(:,3), size(Z_sph));
        h_leg{4} = surf(X_ell, Y_ell, Z_ell, 'FaceColor', VermillionRed, 'EdgeColor', 'none'); alpha(.2)
        
        % extract x,y,z matrices for plotting
        X_ell = reshape(ELLIPSOID_fo(:,1), size(X_sph));
        Y_ell = reshape(ELLIPSOID_fo(:,2), size(Y_sph));
        Z_ell = reshape(ELLIPSOID_fo(:,3), size(Z_sph));
        h_leg{5} = surf(X_ell, Y_ell, Z_ell, 'FaceColor', darkblue, 'EdgeColor', 'none'); alpha(.2)
        
        % extract x,y,z matrices for plotting
        X_ell = reshape(ellipsoid(:,1), size(X_sph));
        Y_ell = reshape(ellipsoid(:,2), size(Y_sph));
        Z_ell = reshape(ellipsoid(:,3), size(Z_sph));
        h_leg{6} = surf(X_ell, Y_ell, Z_ell, 'FaceColor', DupontGray, 'EdgeColor', 'none'); alpha(.2)
    end
    
    plot3(mu(1,4), mu(2,4), mu(3,4), 'o', 'color', [crimson, 0.7], 'MarkerFaceColor', crimson, 'markersize', 8)
    drawnow limitrate
    pause(0.1)
end

legend([h_leg{1}, h_leg{2}, h_leg{3}, h_leg{4}, h_leg{5}, h_leg{6}], ...
    'Nominal path', 'Samples', 'Sample mean', ...
    'Sample covariance - SE(3)', 'First-order covariance - SE(3)', 'Sample covariance - Lie algebra', ...
    'location', 'best', 'fontsize', 14)
axis equal tight
figuresize(21,21,'cm')
print -opengl -dpng -r600 banana_is_gaussian_se3.png




function robot = propagation(robot, u)
% SE(3) propagation model; each input is U \in se(3) plus exp map of the
% noise defined in Lie algebra
Ui = u(1) * robot.G1 + u(2) * robot.G2 + u(3) * robot.G3 + ...
    u(4) * robot.G4 + u(5) * robot.G5 + u(6) * robot.G6;
for i = 1:robot.n
    % sample from a zero mean Gaussian
    noise = robot.L * randn(6,1);
    N = robot.G1 * noise(1) + robot.G2 * noise(2) + robot.G3 * noise(3) + ...
        robot.G4 * noise(4) + robot.G5 * noise(5) + robot.G6 * noise(6);
    robot.x{i} = robot.x{i} * expm(Ui + N);
end
end

function [mu, Sigma] = Lie_sample_statistics(robot)
% compute sample mean and covariance on matrix Lie group
mu0 = robot.x{1};   % pick a sample as initial guess
v = robot.x;
max_iter = 100;
iter = 1;
while iter < max_iter
    mu = mu0 * 0;
    Sigma = zeros(6);
    for i = 1:robot.n
        % left-invariant error: eta^L = X^-1 * X^hat
        v{i} = logm(mu0 \ robot.x{i});
        mu = mu + v{i};
        vec_v = wedge(v{i});
        Sigma = Sigma + vec_v * vec_v';
    end
    mu = mu0 * expm(mu / robot.n);
    Sigma = (1/(robot.n-1)) * Sigma;    % unbiased sample covariance
    % check if we're done here!
    if norm(logm(mu0 \ mu), 'fro') < 1e-8
        return
    else
        mu0 = mu;
    end
    iter = iter + 1;
end
warning('Not converged! max iteration reached. The statistics might not be reliable.')
end


% define SE(3) methods
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
X = [skew(x(4:6)), x(1:3); 0 0 0 0];
end

function x = wedge(X)
% wedge: se(3) -> R^6
x = [X(1:3,4); unskew(X(1:3,1:3))];
end
