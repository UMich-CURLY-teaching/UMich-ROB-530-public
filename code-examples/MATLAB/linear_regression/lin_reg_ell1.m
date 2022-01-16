% Robust linear regression with l1-regularizer
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

% add a few outliers!
outliers = unique(1 + floor(rand(5,1)*length(t))); % random indicies
t(outliers) = t(outliers) + (2*rand(length(outliers),1)-1);

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

% solve for initial w using least squares with l_2 regularizer 
w_init = (Phi' * Phi + 0.1*eye(size(Phi,2))) \ Phi' * t; 

% now solve the ell_1 problem using IRLS
max_iteration = 1000;
iter = 0;
B = eye(N);                 % weights for || t - Phi * w ||^2
G = eye(length(w_init));    % weights for || w ||^2
r = zeros(N,1);
delta = 1e-6;               % to avoid division by zero
eps_termination = 1e-9;     % terminatoin threshold
w = w_init;
w0 = w;
while iter < max_iteration
    iter = iter + 1;
    % compute weights
    G(1,1) = 1 / max(delta, abs(w(1)) );
    for i = 1:N
        B(i,i) = 1 / max(delta, abs(t(i) - Phi(i,:) * w) );
        G(i+1, i+1) = 1 / max(delta, abs(w(i+1)) );
    end
    % normalize weights
    B = B ./ sum(sum(B));
    G = G ./ sum(sum(G));
    
    % solve normal equations
    w = (Phi' * B * Phi + 0.1 * G) \ (Phi' * B * t);
    
    if ~mod(iter,10)
        disp(['iteration: ', num2str(iter)])
        disp(['||w||: ', num2str(norm(w,1))])
    end

    % check if converged
    if norm(w - w0,1) < eps_termination
        disp('Converged')
        break;
    else
        w0 = w;
    end    
end

% find survived basis for plotting, ignoring bias
sb = abs(w(2:end)) > delta;

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
plot(x(sb), t(sb), 'ok', 'markersize', 16, 'linewidth', 3)

% axis([0 8 -1.25 1.25])
axis tight
legend('true process', 'training points', 'prediction', 'survived basis', 'location', 'best')
xlabel('input $x$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('output $y(x)$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
figuresize(21,16,'cm')
% print -opengl -dpng -r600 lin_reg_ell1.png

% plot weights
figure; hold on, grid on
stem(w,'filled', 'markersize', 12, 'linewidth', 1.5)
xlabel('basis', 'fontsize', fsize, 'Interpreter','latex')
ylabel('w', 'fontsize', fsize, 'Interpreter','latex')
axis tight; xticks(1:2:length(w))
set(gca, 'fontsize', fsize)
figuresize(21,16,'cm')
% print -opengl -dpng -r600 lin_reg_w_ell1.png