% An example of the velocity-based motion model on SE(2). The process model 
% is simply X_k+1 = X_k * exp(u_k + w_k) where
% X_k is in SE(2), u_k is the twist in se(2), and w_k is N(0,Q_k) and defined 
% in the Lie algebra se(2). We use Monte Carlo methods to propagate samples 
% over a path and then compute the sample mean and covariance. Note that the 
% sample mean and covariance are computed using an iterative algorithm which 
% is different than usual Euclidean sample statistics. 
% The covariance on Lie algebra is flat as expected but it's nonlinear when 
% mapped to the manifold using Lie exp map. We use expm and logm as
% numerical Lie exp and log map. Both maps have closed-form formulas as
% well.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   01/15/2020

clc; clear; close all

% generate a path
dt = 1;
gt = [];
gt.x = [0:dt:10, 10 * ones(1,length(0:dt:10)), 10:-dt:0, ...
    zeros(1,length(10:-dt:0))];
gt.y = [0 * ones(1,length(0:dt:10)), 0:dt:10, ...
    10 * ones(1,length(10:-dt:0)), 10:-dt:0,];

% find the headings tangent to the path
gt.h = 0;
for i = 2:length(gt.x)
   gt.h(i) = atan2(gt.y(i) - gt.y(i-1), gt.x(i) - gt.x(i-1));
end
% ground truth poses
H = cell(length(gt.x),1);
H{1} = eye(3);
for i = 2:length(gt.x)
    H{i} = posemat(gt.x(i), gt.y(i), gt.h(i));
end
% generate noise-free twist control inputs (velocity commands) in the Lie algebra
u = cell(length(gt.x),1);
u{1} = zeros(3);
for i = 2:length(gt.x)
    u{i} = logm(H{i-1} \ H{i});
end

% construct noise free motion trajectory (sanity check for the generated
% inputs!)
path = [];
path.T = H{1};
path.x = 0;
path.y = 0;
for i = 2:size(u,1)
    path.T = path.T * expm(u{i});
    path.x(i+1) = path.T(1,3);
    path.y(i+1) = path.T(2,3);
end


% build a 2D robot! this should be a class so we can easily create an 
% object but I use a simple strcuture here.
robot = [];
robot.dt = dt;
robot.x = eye(3);       % state mean
robot.Cov = zeros(3);   % covariance
% motion model noise covariance
robot.Q = diag([0.01^2, 0.01^2, 0.01^2]);
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

fsize = 30; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

h_leg = []; % legend handle
figure; hold on
h_leg{1} = plot(path.x, path.y, '-', 'color', Darkgrey, 'linewidth', 3);
grid on, axis auto equal, axis([-6 6 -1 5])
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
axis auto equal

% plot initial mean 
h_leg{2} = plot(robot.x(1,3), robot.x(2,3),'.','Color', [green, .25], 'markersize', 14);

% main loop; iterate over the control inputs and move the robot
ELLIPSE = zeros(size(circle,1),2); % covariance ellipse on manifold (nonlinear)
ellipse = zeros(size(circle,1),2); % covariance ellipse on Lie algebra

for i = 1:size(u,1)
    % move th robot based on the input
    robot = propagation(robot, u{i});
    
    robot.L = chol(robot.Cov, 'lower');
    for j = 1:size(circle,1)
        % sample covariance on SE(2)
        ell_se2_vec = scale * robot.L * circle(j,:)';
        % retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
        temp = robot.x * expm(robot.G1 * ell_se2_vec(1) + robot.G2 * ell_se2_vec(2) + robot.G3 * ell_se2_vec(3));
        ELLIPSE(j,:) = [temp(1,3), temp(2,3)];
        
    end
    % plot the ellipses
    h_leg{3} = plot(ELLIPSE(:,1), ELLIPSE(:,2), 'color', [VermillionRed, 0.7], 'linewidth', 2);
%     h_leg{4} = plot(ellipse(:,1), ellipse(:,2), 'color', [DupontGray, 0.7], 'linewidth', 2);
    %     end
    plot(robot.x(1,3), robot.x(2,3), 'o', 'color', [crimson, 0.7], 'MarkerFaceColor', crimson, 'markersize', 8)
    drawnow limitrate
    pause(0.1)
end

legend([h_leg{1}, h_leg{2}, h_leg{3}], ...
    'Nominal path', 'Mean', 'Covariance - SE(2)', 'location', 'best', 'fontsize', 14)
axis auto equal
% figuresize(21,21,'cm')
% print -opengl -dpng -r600 banana_is_gaussian.png



function H = posemat(x,y,h)
% construct a SE(2) matrix element
H = [cos(h) -sin(h) x;
     sin(h) cos(h) y; 
     0 0 1];
end


function robot = propagation(robot, u)
% SE(2) propagation model; the input is u \in se(2) plus noise
    % sample from a zero mean Gaussian 
    noise = robot.L * randn(3,1);
    N = robot.G1 * noise(1) + robot.G2 * noise(2) + robot.G3 * noise(3);
    xi = u + N;
    % propagate mean
    robot.x = robot.x * expm(xi);
    % propagate covariance
    robot.Cov = robot.Ad(expm(-xi)) * robot.Cov * robot.Ad(expm(-xi))' ...
        + robot.Q;
end