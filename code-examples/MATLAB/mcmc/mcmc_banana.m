% MCMC (random-walk Metropolis-Hastings) example for sampling from a banana distribution
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
x = zeros(2,1);
prop_cov = diag([1, 1]); 
prop_L = chol(prop_cov, 'lower');
sample = x;

for i = 1:Max_iter
   % sample from the proposal distribution to create a Markov chain
   x_new = x + prop_L * randn(2,1);
   
   % accept with probability A
   u = rand;
   A = min(1, f(x_new(1), x_new(2)) / f(x(1),x(2)) );
   if A > u % accept
       x = x_new;
       sample = [sample, x];
       accepted = accepted + 1;
   else
       rejected = rejected + 1;
   end    
end


% plot
figure;
hist3(sample','Nbins',[30 30],'CDataMode','auto','FaceColor','interp')
view(0,90)

disp(['Accept ratio:  ', num2str(accepted/Max_iter)])
disp(['Reject ratio:  ', num2str(rejected/Max_iter)])
