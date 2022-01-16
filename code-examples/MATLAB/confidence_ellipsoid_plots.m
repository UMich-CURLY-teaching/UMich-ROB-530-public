%{  
    Copyright (C) 2017  Maani Ghaffari Jadidi
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

fsize = 24; % font size
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

%% 2D Example
% MVN
mu = [0; 0.5]; % mean
Sigma = [.8 -0.3; -0.3 1]; % covariance

% Draw samples
d = 2; % dimension of data
n = 1000; % number of samples
X = zeros(d,n);
L = chol(Sigma, 'lower'); % Cholesky factor of covariance
for i = 1:n
    Z = randn(d,1); % draw a d-dimensional vector of standard normal random variables
    X(:,i) = L * Z + mu;
end
X = X'; % make rows to be samples/observations

% Sample mean
mu_bar = (1/n * sum(X))'; % transpose to make it a column vector

disp(['    mean  ', '   sample mean'])
disp([mu, mu_bar])

% Sample covariance
e = (X - mu_bar')'; % centralize samples around sample mean
Sigma_bar = (e * e') / (n-1);
% Alternative option using MATLAB cov()
% Sigma_bar = cov(X);

disp(['    covariance  ', '       sample covariance'])
disp([Sigma, Sigma_bar])

% create confidence ellipse
% first create points from a unit circle
phi = (-pi:.01:pi)';
circle = [cos(phi), sin(phi)];

% Chi-squared 2-DOF 95% percent confidence (0.05): 5.991
scale = sqrt(5.991);
% apply the transformation and scale of the covariance
ellipse = (scale * L * circle' + mu)';
% test plot to visualize circle and ellipse together: 
% figure; hold on; 
% plot(circle(:,1), circle(:,2))
% plot(ellipse(:,1), ellipse(:,2))
% plot(mu(1), mu(2), 's', 'markersize', 12)
% axis equal auto, grid on 


figure; hold on
h = []; % plot handle
h{1} = plot(X(:,1), X(:,2), '.', 'color', darkblue, 'markersize', 8);
h{2} = plot(ellipse(:,1), ellipse(:,2), 'color', VermillionRed, 'linewidth', 2);
h{3} = plot(mu_bar(1), mu_bar(2), 'sk', 'MarkerFaceColor', 'k', 'markersize', 10);
xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex');
legend([h{1}, h{2}, h{3}], 'Samples', '$95\%$ Confidence Ellipse', 'Sample Mean', 'location', 'best')
text(-0.5, 3.65, '$\mathcal{N}([0;0.5], [0.8,-0.3;-0.3,1])$', 'fontsize',20, 'Interpreter','latex')
axis([-4 4 -4 4]), grid on
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
% print -painters -dpdf -r600 confidence_ellipse.pdf


%% 3D Example
% MVN
mu = [0; 0.5; 1]; % mean
Sigma = [0.8 -0.3  0.1;
        -0.3  1.0 -0.2
         0.1  -0.2 0.5]; % covariance

% Draw samples
d = 3; % dimension of data
n = 1000; % number of samples
X = zeros(d,n);
L = chol(Sigma, 'lower'); % Cholesky factor of covariance
for i = 1:n
    Z = randn(d,1); % draw a d-dimensional vector of standard normal random variables
    X(:,i) = L * Z + mu;
end
X = X'; % make rows to be samples/observations

% Sample mean
mu_bar = (1/n * sum(X))'; % transpose to make it a column vector

disp(['    mean  ', '   sample mean'])
disp([mu, mu_bar])

% Sample covariance
e = (X - mu_bar')'; % centralize samples around sample mean
Sigma_bar = (e * e') / (n-1);
% Alternative option using MATLAB cov()
% Sigma_bar = cov(X);

disp(['    covariance  ', '                 sample covariance'])
disp([Sigma, Sigma_bar])

% create confidence ellipsoid
% first create points from a unit sphere
phi = linspace(-pi, pi, 1000)';
theta = linspace(-pi/2, pi/2, 1000)';
[PHI, THETA] = meshgrid(phi, theta);
X_sph = cos(THETA) .* cos(PHI);
Y_sph = cos(THETA) .* sin(PHI);
Z_sph = sin(THETA);
sphere = [X_sph(:), Y_sph(:), Z_sph(:)];

% Chi-squared 3-DOF 95% percent confidence (0.05): 7.815
scale = sqrt(7.815);
% apply the transformation and scale of the covariance
ellipsoid = (scale * L * sphere' + mu)';
% extract x,y,z matrices for plotting
X_ell = reshape(ellipsoid(:,1), size(X_sph));
Y_ell = reshape(ellipsoid(:,2), size(Y_sph));
Z_ell = reshape(ellipsoid(:,3), size(Z_sph));
% test plot to visualize sphere and ellipsoid together: 
% figure; hold on; 
% surf(X_sph, Y_sph, Z_sph, 'FaceColor', green, 'EdgeColor', 'none'), alpha(.4), colormap jet
% surf(X_ell, Y_ell, Z_ell, 'FaceColor', VermillionRed, 'EdgeColor', 'none'), alpha(.4)
% axis equal auto, grid on 

figure; hold on
view(145, 30)
h = []; % plot handle
h{1} = plot3(X(:,1), X(:,2), X(:,3), '.', 'color', darkblue, 'markersize', 10);
h{2} = surf(X_ell, Y_ell, Z_ell, 'FaceColor', VermillionRed, 'EdgeColor', 'none'); alpha(.3)
h{3} = plot3(mu_bar(1), mu_bar(2), mu_bar(3), 'sk', 'MarkerFaceColor', 'k', 'markersize', 14);
xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex');
zlabel('$x_3$', 'Interpreter','latex');
legend([h{1}, h{2}, h{3}], 'Samples', '$95\%$ Confidence Ellipsoid', 'Sample Mean', 'location', 'best')
axis equal tight, grid on
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
print -opengl -dpng -r600 confidence_ellipsoid_3d.png
