% Linear regression with l2-regularizer
%
%   Author: Maani Ghaffari Jadidi
%   Date:   03/22/2020

clc; clear; close all

% create a dataset
z = (0:0.1:8)'; 
y = exp(-0.1*z) .* cos(z); % true process

% pick some training points randomly
N = 15; % number of training points
idx = unique(1 + floor(rand(N,1)*length(z))); % random indicies
N = length(idx);
x = z(idx); % training input
t = y(idx) + 0.05 * randn(N,1); % noisy target 

% here we solve for w in a linear model y = w^T * phi
s = 1.75; % bandwidth (length-scale)
basis = @(x1,x2) exp(- norm(x1 - x2).^2 / (2*s^2) );

% compute basis (design) matrix
Phi = zeros(length(x), length(x)+1);
Phi(:,1) = 1; % bias basis
for i = 1:size(Phi,1)
    for j = 1:size(Phi,2)-1
        Phi(i,j+1) = basis(x(i), x(j));
    end
end

% solve for w using least squares with l_2 regularizer 
w = (Phi' * Phi + 0.1*eye(size(Phi,2))) \ (Phi' * t); 

% predict at arbitrary inputs
Phi_test = zeros(length(z), length(w));
Phi_test(:,1) = 1; % bias basis
for i = 1:size(Phi_test,1)
    for j = 1:size(Phi_test,2)-1
        Phi_test(i,j+1) = basis(z(i), x(j));
    end
end

y_test = Phi_test * w; % predict all at once


% plotting
fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

% plot the true process and training points
figure; hold on, grid on
plot(z, y, x, t, '.', 'linewidth', 3, 'markersize', 24)
% update the plot with predictions
plot(z, y_test, '--k', 'linewidth', 3)

axis([0 8 -1.25 1.25])
legend('true process', 'training points', 'prediction', 'location', 'best')
xlabel('input $x$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('output $y(x)$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
figuresize(21,16,'cm')
% print -opengl -dpng -r600 lin_reg.png

% plot weights
figure; hold on, grid on
stem(w,'filled', 'markersize', 12, 'linewidth', 1.5)
xlabel('basis', 'fontsize', fsize, 'Interpreter','latex')
ylabel('w', 'fontsize', fsize, 'Interpreter','latex')
axis tight; xticks(1:2:length(w))
set(gca, 'fontsize', fsize)
figuresize(21,16,'cm')
% print -opengl -dpng -r600 lin_reg_w.png