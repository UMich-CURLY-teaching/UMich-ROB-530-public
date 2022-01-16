% An example for showing the relation between SO(3) Lie algebra and its
% generators with the usual cross product on R^3.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   02/13/2019

clc; clear; close all

% create random vectors and their cross product
a = floor(4*rand(3,1)-2);
b = floor(4*rand(3,1)-2);
c1 = cross(a,b);

% vector to skew R^3 -> so(3)
skew = @(x) [   0,  -x(3),  x(2);
             x(3),      0,  -x(1);
             -x(2), x(1),   0];
% so(3) -> R^3
unskew = @(X) [X(3,2); X(1,3); X(2,1)];
         
% R^3 standard basis
e1 = [1;0;0];
e2 = [0;1;0];
e3 = [0;0;1];

% so(3) (SO(3) Lie algebra) infinitesimal generators
G1 = skew(e1);
G2 = skew(e2);
G3 = skew(e3);

% reproduce the same cross product using skew(a) * b  = a x b
c2 = skew(a) * b;

% now try the application of each generator independently and sum them!
c3 = a(1)* G1 * b + a(2) * G2 * b + a(3) * G3 * b;

disp('     a x b       skew(a) * b       sum(a(i) * G_i * b)')
for i = 1:3
    % to adjust the spacing for negative sign.
    if c1(i) < 0
        disp(['      ', num2str(c1(i)), '              ', num2str(c2(i)), '                ', num2str(c3(i)), '   '])
    else
        disp(['       ', num2str(c1(i)), '               ', num2str(c2(i)), '                 ', num2str(c3(i)), '   '])
    end
end


% now let's try playing with the Lie bracket! we take the Lie bracket of
% any two generators and get the third generator back. This is similar
% cross product of any two R^3 standard basis.
bracket = @(A,B) A * B - B * A;

fprintf('\n-----------------------\n\n')

disp('unskew(bracket(G1,G2)) - cross(e1,e2) = ')
disp(unskew(bracket(G1,G2)) - cross(e1,e2))

disp('unskew(bracket(G2,G3)) - cross(e2,e3) = ')
disp(unskew(bracket(G2,G3)) - cross(e2,e3))

disp('unskew(bracket(G3,G1)) - cross(e3,e1) = ')
disp(unskew(bracket(G3,G1)) - cross(e3,e1))


