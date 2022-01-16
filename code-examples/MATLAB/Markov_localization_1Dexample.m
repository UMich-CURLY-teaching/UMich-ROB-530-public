%{  
    Copyright (C) 2018  Maani Ghaffari Jadidi
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details. 
%}

% Markov localization example in a 1D world.
clc; clear; close all

fsize = 14; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

% colours
green = [0.2980 .6 0];
darkblue = [0 .2 .4];
VermillionRed = [156,31,46]/255;

%% Action set; 0: stay, 1: move forward
% control inputs (sequence of actions)
u = [1 1 1 1 1 1 1 0];

%% Measurements
z = [1 0 0 0 0 1 0 0];

%% State space: the world has 20 cells, after 20 the robot will be at cell 1 again.
X = 1:20;

%% Belief initialization 
bel = ones(1,20) * 1/20; % uniform prior

% plot prior belief
figure; figuresize(21,21,'cm')
subplot(3,1,1); hold on; grid on
title('Prior Belief Map')
bar(bel, 'FaceColor', darkblue)
axis([0.5 20.5 0 1])
ylabel('$p(x)$', 'Interpreter','latex');
set(gca,'XTick', 1:20)
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')

%% Likelihood map to provide measurements
% The robot receives measurements at cell 4, 9, and 13
likelihood_map = ones(1,20) * .2;
likelihood_map([4,9,13]) = .8;

% plot likelihood
subplot(3,1,2); hold on; grid on
title('Likelihood Map')
bar(likelihood_map, 'FaceColor', green)
axis([0.5 20.5 0 1])
ylabel('$p(z|x)$', 'Interpreter','latex');
set(gca,'XTick', 1:20)
set(gca,'fontsize',fsize)
set(gca,'TickLabelInterpreter','latex')


%% Markov localization using Bayes filter

% The main loop can be run forever, but we run it for a limited sequence of
% control inputs.
k = 1; % step counter
bel_predicted = bel;   % predicted belief
while ~isempty(u)
    
    if z(k) == 1 % measurement received
        eta = 0;    % normalization constant
        for i = 1:length(X)
            likelihood = measurement_model(X(i), likelihood_map); % get measurement likelihood
            bel(i) = likelihood * bel_predicted(i); % unnormalized Bayes update
            eta = eta + bel(i);
        end
        bel = bel ./ eta; % normalize belief
    end
    
    % prediction; belief convolution
    for i = 1:length(X)
        bel_predicted(i) = 0;
        for j = 1:length(X)
            pu = motion_model(X(i), X(j), u(1));
            bel_predicted(i) = bel_predicted(i) + pu * bel(j);
        end
    end
    
    % set the predicted belief as prior
    bel = bel_predicted;
    
    % remove the executed action from the list
    u(1) = [];
    k = k + 1;
    
    % plot posterior belief
    subplot(3,1,3); grid on
    bar(bel, 'FaceColor', VermillionRed)
    axis([0.5 20.5 0 1])
    ylabel('$p(x|z)$', 'Interpreter','latex');
    set(gca,'XTick', 1:20)
    set(gca,'fontsize',fsize)
    set(gca,'TickLabelInterpreter','latex')
    title('Posterior Belief Map')
    
    drawnow;
    pause(.2)
end


function p = motion_model(xi, xj, u)
% Motion model changes the belief based on action u
if u == 1 % move forward
    dx = xi-xj;
    if dx == 1 || dx == -19
        p = 1;
    else 
        p = 0;
    end
elseif u == 0 % stay
    dx = xi-xj;
    if dx == 0
        p = 1;
    else
        p = 0;
    end
else
    assert(u == 1 || u == 0, 'The action is not defined')
end
end
    
function p = measurement_model(x, likelihood_map)
% measurement model returns p(z|x) based on a likelihood map.
p = likelihood_map(x);
end
    