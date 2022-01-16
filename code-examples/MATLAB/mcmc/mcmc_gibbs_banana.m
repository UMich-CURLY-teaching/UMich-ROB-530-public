% MCMC (Gibbs sampling) example for sampling from a banana distribution
% http://www.stat.columbia.edu/~gelman/research/published/GelmanMeng1991
%   Author: Maani Ghaffari Jadidi
%   Date:   04/27/2020

clc; clear; close all



% banana distribution, play with A, B, C1, and C2 to get different shapes
A = 5; B = 1; C1 = 4; C2 = 4;
f = @(x1,x2) exp(-0.5 * (A * x1.^2 .* x2.^2 + x1.^2 + x2.^2 ...
    - 2 * B * x1 .* x2 - 2 * C1 * x1 - 2 * C2 * x2) );

[X1,X2] = meshgrid(-1:0.01:6);
Z = f(X1,X2);
figure;
pcolor(X1,X2,Z), shading interp
% contour(X1,X2,Z)


% MCMC 
rejected = 0;
accepted = 0;
Max_iter = 100000;

% proposal distribution
x2 = randn;
sample = [];

for i = 1:Max_iter
   % sample from the proposal distribution to create a Markov chain
   % sample from p(x1)
   x1 = (B * x2 + C1) / (A * x2^2 + 1) +  sqrt(1 / (A * x2^2 + 1)) * randn; 
   % sample from p(x2|x1)
   x2 = (B * x1 + C2) / (A * x1^2 + 1) +  sqrt(1 / (A * x1^2 + 1)) * randn; 
   
   sample = [sample, [x1; x2] ];
end


% plot
figure;
hist3(sample','Nbins',[30 30],'CDataMode','auto','FaceColor','interp')
view(0,90)

