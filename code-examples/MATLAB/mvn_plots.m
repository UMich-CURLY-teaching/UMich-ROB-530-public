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

% MVN
mu = [0 0.5]; % mean
Sigma = [.8 0.3; 0.3 1]; % covariance
% compute PDF
x1 = -3:.1:3; x2 = -3:.1:4;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));

figure;
surf(x1,x2,F); shading flat
axis([-3 3 -3 4 0 .2])
xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex');
zlabel('PDF', 'Interpreter','latex');
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
print -opengl -dpng -r300 mvn_surface.png

% marginals
f1 = normpdf(x1, mu(1), Sigma(1,1));
f2 = normpdf(x2, mu(2), Sigma(2,2));
% x_1 conditioned on x_2
mu_12 = mu(1) + Sigma(1,2) * (1/Sigma(2,2)) * (0.9 - 0.5); % conditional mean
sigma_12 = Sigma(1,1) - Sigma(1,2) * (1/Sigma(2,2)) * Sigma(2,1);  % conditional variance
f12 = normpdf(x1, mu_12, sigma_12);
% marginal plot
figure; hold on
h = []; % plot handle
h{1} = plot(x1, f1, 'linewidth', 2);
h{2} = plot(x2, f2, '--', 'linewidth', 2);
h{3} = plot(x1, f12, '-.', 'linewidth', 2);
axis auto, grid on
xlabel('$x$', 'Interpreter','latex'); 
ylabel('PDF', 'Interpreter','latex');
legend([h{1}, h{2}, h{3}], '$p(x_1)$', '$p(x_2)$', '$p(x_1|x_2 = 0.9)$', 'location', 'northeast')
text(-3.95, 0.58, '$p(x_1) = \mathcal{N}(0,0.8)$', 'fontsize',16, 'Interpreter','latex')
text(-3.95, 0.54, '$p(x_2) = \mathcal{N}(0.5,1)$', 'fontsize',16, 'Interpreter','latex')
text(-3.95, 0.51, '$p(x_1|x_2 = 0.9) = \mathcal{N}(0.12,0.71)$', 'fontsize',16, 'Interpreter','latex')
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
print -painters -dpdf -r600 mvn_marginals_cond.pdf

figure;
pcolor(x1,x2,F);
shading interp, grid on
axis([-3 3 -3 4])
xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex');
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
print -opengl -dpng -r300 mvn_top.png

figure;
contour(x1,x2,F, 'ShowText','on')
axis tight, grid on
xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex');
text(-2.5, 3.25, '$\mathcal{N}([0;0.5], [0.8,0.3;0.3,1])$', 'fontsize',fsize, 'Interpreter','latex')
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')
figuresize(21,21,'cm')
print -painters -dpdf -r600 mvn_contour.pdf
