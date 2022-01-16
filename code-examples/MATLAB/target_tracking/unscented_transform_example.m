% An example of using the unscented transform for propagating a Gaussian
% distribution throuh a nonlinear map. We use polar to Cartesian for this
% code.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   01/16/2020

clc; clear; close all

% creat a random mean and covariance
x = [1.5; pi/6]; % x = (r, theta)
% P = [0.3^2, -0.14^2; -0.14^2, 0.35^2];
P = [0.1^2, -0.09^2; -0.09^2, 0.6^2];

% build a nonlinear map
f = @(x) [x(1) * cos(x(2)); x(1) * sin(x(2))];

% propagate the uncertainty using UT and affine model to compare
kappa = 2;
ut = unscented_transform(x, P, f, kappa);
ut.propagate();

% visualization
fsize = 22; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
green = [0.2980 .6 0];
% crimson = [220,20,60]/255; 
% darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
% darkgrey = [.35 .35 .35];
% lightgrey = [.7 .7 .7];
% Lightgrey = [.9 .9 .9];
VermillionRed = [156,31,46]/255;
DupontGray = [144,131,118]/255;
% Azure = [53, 112, 188]/255;
% purple = [178, 102, 255]/255;
% orange = [255,110,0]/255;

% create confidence ellipse
% first create points from a unit circle
phi = (-pi:.01:pi)';
circle = [cos(phi), sin(phi)];
% Chi-squared 2-DOF 95% percent confidence (0.05): 5.991
scale = sqrt(5.991);
% apply the transformation and scale of the covariance
ellipse_polar = (scale * chol(P,'lower') * circle' + x)';
ellipse_cartesian = (scale * chol(ut.Cov,'lower') * circle' + ut.mean)';

% generate samples for both polar and cartesian coordinates
s_polar = (chol(P,'lower') * randn(2,1000) + x)';
s_cartesian = zeros(size(s_polar));
for i = 1:size(s_polar,1)
    s_cartesian(i,:) = f(s_polar(i,:));
end

% plot in polar coordinates
figure; hold on; grid on
h = []; % plot handle
h{1} = plot(s_polar(:,1), s_polar(:,2), '.', 'color', DupontGray);
h{2} = plot(x(1), x(2), 'o', 'color', VermillionRed, 'markersize', 18);
h{3} = plot(ellipse_polar(:,1), ellipse_polar(:,2), 'color', VermillionRed, 'linewidth', 3);
h{4} = plot(ut.X(1,:), ut.X(2,:), '.', 'color', Darkgrey, 'markersize', 32);
xlabel('$r$', 'Interpreter','latex'); 
ylabel('$\theta$', 'Interpreter','latex');
legend([h{1}, h{2}, h{3}, h{4}], 'Samples', 'Mean', '$95\%$ Confidence Ellipse', 'Sigma Points', 'location', 'north outside')
text(1.75, 1.6, '$\kappa = 2$', 'fontsize',fsize, 'Interpreter','latex')
axis equal auto
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
% print -opengl -dpng -r600 ut_example_polar.png

% plot in Cartesian coordinates
figure; hold on; grid on
h = []; % plot handle
h{1} = plot(s_cartesian(:,1), s_cartesian(:,2), '.', 'color', DupontGray);
h{2} = plot(ut.mean(1), ut.mean(2), 'o', 'color', VermillionRed, 'markersize', 18);
h{3} = plot(ellipse_cartesian(:,1), ellipse_cartesian(:,2), 'color', VermillionRed, 'linewidth', 3);
h{4} = plot(ut.Y(1,:), ut.Y(2,:), '.', 'color', Darkgrey, 'markersize', 32);
xlabel('$x=r\cos(\theta)$', 'Interpreter','latex'); 
ylabel('$y=r\sin(\theta)$', 'Interpreter','latex');
legend([h{1}, h{2}, h{3}, h{4}], 'Samples', 'Mean', '$95\%$ Confidence Ellipse', 'Sigma Points', 'location', 'north outside')
text(1.6, 1.8, '$\kappa = 2$', 'fontsize',fsize, 'Interpreter','latex')
axis equal auto
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
% print -opengl -dpng -r600 ut_example_cart.png
