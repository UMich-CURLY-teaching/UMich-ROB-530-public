% An example of process models on Lie groups and uncertainty propagation
% using SE(2). The process model is simply X_k+1 = X_k * U_k exp(w_k) where
% X_k, U_k are both in SE(2) and w_k is N(0,Q_k) and defined in the Lie
% algebra se(2). We use Monte Carlo methods to propagate samples over a
% path and then compute the sample mean and covariance on Lie group, here 
% SE(2). Note that the sample mean and covariance are computed using an 
% iterative algorithm which is different than usual Euclidean sample 
% statistics. The covariance on Lie algebra is flat as expected but it's
% nonlinear when mapped to the manifold using Lie exp map.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   02/14/2019
%   Happy Valentine's Day, implemented with love and Opeth:
%   https://www.youtube.com/watch?v=4oWhkjQWE24

clc; clear; close all

% generate a path
dt = 0.6;
gt = [];
gt.x = 0:dt:4;
gt.y = 0.1 * exp(0.6 * gt.x) - 0.1;

% find the headings tangent to the path
gt.h = 0;
for i = 2:length(gt.x)
   gt.h(i) = atan2(gt.y(i) - gt.y(i-1), gt.x(i) - gt.x(i-1));
end

% generate noise-free control inputs
u = [];
u(1,:) = diff(gt.x);
u(2,:) = diff(gt.y);
u(3,:) = diff(gt.h);

% build a 2D robot!
robot = [];
robot.dt = dt;
robot.n = 1000;
robot.x = cell(robot.n,1); % state mean
robot.Cov_fo = zeros(3); % first order covariance propagation around mean
for i = 1:robot.n
    robot.x{i} = eye(3);
end
% motion model noise covariance
robot.Q = diag([0.03^2, 0.03^2, 0.1^2]);
% Cholesky factor of covariance for sampling
robot.L = chol(robot.Q, 'lower');
% se(2) generators; twist = vec(v1, v2, omega).  
robot.G1 = [0     0     1
            0     0     0
            0     0     0];

robot.G2 = [0     0     0
            0     0     1
            0     0     0];

robot.G3 = [0    -1     0
            1     0     0
            0     0     0];
% SE(2) Adjoint
robot.Ad = @(X) [X(1:2,1:2), [X(2,3); -X(1,3)]; 0 0 1];

% construct noise free motion trajectory
path = [];
path.T = eye(3);
path.x = 0;
path.y = 0;
for i = 1:size(u,2)
    Ui = [cos(u(3,i)) -sin(u(3,i)) u(1,i);
         sin(u(3,i)) cos(u(3,i)) u(2,i); 
         0 0 1];
    path.T = path.T * Ui;
    path.x(i+1) = path.T(1,3);
    path.y(i+1) = path.T(2,3);
end

% create confidence ellipse
% first create points from a unit circle + angle (third dimension of so(3))
phi = (-pi:.01:pi)';
circle = [cos(phi), sin(phi), zeros(length(phi),1)];
% Chi-squared 3-DOF 95% percent confidence (0.05): 7.815
scale = sqrt(7.815);

% incremental visualization
green = [0.2980 .6 0];
crimson = [220,20,60]/255; 
darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
% darkgrey = [.35 .35 .35];
% lightgrey = [.7 .7 .7];
% Lightgrey = [.9 .9 .9];
VermillionRed = [156,31,46]/255;
DupontGray = [144,131,118]/255;
% Azure = [53, 112, 188]/255;
% purple = [178, 102, 255]/255;
% orange = [255,110,0]/255;

fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

h_leg = []; % legend handle
figure; hold on
h_leg{1} = plot(path.x, path.y, '-', 'color', Darkgrey, 'linewidth', 3);
grid on, axis auto equal, axis([-6 6 -1 5])
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
axis tight equal

% extract propagated particles
p = [];
for i = 1:robot.n
    p(1,i) = robot.x{i}(1,3);
    p(2,i) = robot.x{i}(2,3);
end

% compute sample statistics
[mu, Sigma] = Lie_sample_statistics(robot);

% plot sample mean and particles
h_leg{2} = plot(p(1,:), p(2,:),'.','Color', [green, .25], 'markersize', 14);
h_leg{3} = plot(mu(1,3), mu(2,3), 'o', 'color', crimson, 'MarkerFaceColor', crimson, 'markersize', 8);

% main loop; iterate over the control inputs and move the robot
ELLIPSE = zeros(size(circle,1),2); % covariance ellipse on manifold (nonlinear)
ELLIPSE_fo = zeros(size(circle,1),2); % first order covariance ellipse on manifold (nonlinear)
ellipse = zeros(size(circle,1),2); % covariance ellipse on Lie algebra

for i = 1:size(u,2)
    % move particles based on the input
    robot = propagation(robot, u(:,i));
    % extract propagated particles
    p = [];
    for j = 1:robot.n
        p(1,j) = robot.x{j}(1,3);
        p(2,j) = robot.x{j}(2,3);
    end
    
    % show particles
    set(h_leg{2},'XData', p(1,:),'YData', p(2,:));
    
    % compute sample statistics
    [mu, Sigma] = Lie_sample_statistics(robot);
    
    % compute first order analytical covariance propagation
     Ui = [cos(u(3,i)) -sin(u(3,i)) u(1,i);
         sin(u(3,i)) cos(u(3,i)) u(2,i); 
         0 0 1];
    
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
        for j = 1:size(circle,1)
        % sample covariance on SE(2)
        ell_se2_vec = scale * L * circle(j,:)';
        % retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
        temp = mu * expm(robot.G1 * ell_se2_vec(1) + robot.G2 * ell_se2_vec(2) + robot.G3 * ell_se2_vec(3));
        ELLIPSE(j,:) = [temp(1,3), temp(2,3)];
        
        % get the ellipse on Lie algebra
        temp = mu * [cos(ell_se2_vec(3)) -sin(ell_se2_vec(3)) ell_se2_vec(1);
                         sin(ell_se2_vec(3)) cos(ell_se2_vec(3)) ell_se2_vec(2); 
                         0 0 1];
        ellipse(j,:) = [temp(1,3), temp(2,3)];
        
        % sample covariance on SE(2)
        ell_se2_vec = scale * robot.L_fo * circle(j,:)';
        % retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
        temp = mu * expm(robot.G1 * ell_se2_vec(1) + robot.G2 * ell_se2_vec(2) + robot.G3 * ell_se2_vec(3));
        ELLIPSE_fo(j,:) = [temp(1,3), temp(2,3)];
        end
        % plot the ellipses
        h_leg{4} = plot(ELLIPSE(:,1), ELLIPSE(:,2), 'color', [VermillionRed, 0.7], 'linewidth', 2);
        h_leg{5} = plot(ELLIPSE_fo(:,1), ELLIPSE_fo(:,2), 'color', [darkblue, 0.7], 'linewidth', 2);
        h_leg{6} = plot(ellipse(:,1), ellipse(:,2), 'color', [DupontGray, 0.7], 'linewidth', 2);
    end
    plot(mu(1,3), mu(2,3), 'o', 'color', [crimson, 0.7], 'MarkerFaceColor', crimson, 'markersize', 8)
    drawnow limitrate
    pause(0.1)
end

legend([h_leg{1}, h_leg{2}, h_leg{3}, h_leg{4}, h_leg{5}, h_leg{6}], ...
    'Nominal path', 'Samples', 'Sample mean', ...
    'Sample covariance - SE(2)', 'First-order covariance - SE(2)', 'Sample covariance - Lie algebra', ...
    'location', 'best', 'fontsize', 14)
axis equal tight
% figuresize(21,21,'cm')
% print -opengl -dpng -r600 banana_is_gaussian.png




function robot = propagation(robot, u)
% SE(2) propagation model; each input is U \in SE(2) plus exp map of the
% noise defined in Lie algebra
for i = 1:robot.n
    % sample from a zero mean Gaussian 
    noise = robot.L * randn(3,1);
    N = robot.G1 * noise(1) + robot.G2 * noise(2) + robot.G3 * noise(3);
    Ui = [cos(u(3)) -sin(u(3)) u(1);
         sin(u(3)) cos(u(3)) u(2); 
         0 0 1];
    robot.x{i} = robot.x{i} * Ui * expm(N);
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
    Sigma = zeros(3);
    for i = 1:robot.n
        % left-invariant error: eta^L = X^-1 * X^hat
        v{i} = logm(mu0 \ robot.x{i});
        mu = mu + v{i};
        vec_v = [v{i}(1,3); v{i}(2,3); v{i}(2,1)];
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