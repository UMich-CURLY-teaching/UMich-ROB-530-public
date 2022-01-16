% MCMC (Hamiltonian Monte Carlo) example for sampling from a banana distribution
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


% HMC 
% We construct the Hamintonian; The total energy of the system is the
% sum of its potential, E(z) = -log(f(z(1),z(2))), and kinetic, K(r) =
% 0.5*r'*r , energies.
H = @(z,r) -log(f(z(1),z(2))) + 0.5*r'*r;

% We need the Jacobian of E for the leapfrog integration
% syms x1 x2 A B C1 C2 real
% E = (A * x1.^2 .* x2.^2 + x1.^2 + x2.^2 - 2 * B * x1 .* x2 - 2 * C1 * x1 - 2 * C2 * x2);
% J = jacobian(E,[x1;x2])';
J = @(x1,x2) [2*A*x1*x2^2 - 2*B*x2 - 2*C1 + 2*x1; 2*A*x2*x1^2 - 2*B*x1 - 2*C2 + 2*x2];

rejected = 0;
accepted = 0;
Max_iter = 100000;

% proposal distribution
r = zeros(2,1);
prop_cov = diag([1, 1]); 
prop_L = chol(prop_cov, 'lower');
x = zeros(2,1);
sample = [];

for i = 1:Max_iter
   % draw a random momentom from the proposal distribution
   r_new = r + prop_L * randn(2,1);
   % take a leapfrog step
   [x_new, r_new] = leapfrog_integration(x, r_new, J);
   
   % accept with probability A
   u = rand;
   A = min(1, exp(H(x, r) - H(x_new, r_new)));
   if A > u % accept
       x = x_new;
       r = r_new;
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


function [z_new, r_new] = leapfrog_integration(z,r, J)
e = 0.1;
r_half = r - e/2 * J(z(1),z(2));
z_new = z + e * r_half;
r_new = r_half - e/2 * J(z_new(1),z_new(2));
end
