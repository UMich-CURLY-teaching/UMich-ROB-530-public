clear all;
more on;

disp('random measurement matrix');
A = rand(10,4)

disp('QR factorization using Givens rotations');
[Q,R] = QR(A)

disp('note that R is indeed equal to the information matrix square root (up to sign flip of rows)')
R_chol = chol(A'*A, 'upper')

disp('here is the transformed measurement matrix');
At = Q'*A;
At(abs(At)<abs(2*eps)) = 0

disp('suppose now that we add a new measurement row');
At_aug = [At; 0 0 0 1]

disp('we can incrementally obtain the new factorization using one Givens rotation');
alpha = At_aug(4,4);
beta = At_aug(11,4);
if beta == 0
    disp('beta = 0');
    c = 1; s = 0;
elseif abs(beta) > abs(alpha)
    disp('|beta| > |alpha|');
    c = -alpha/beta/sqrt(1+(alpha/beta)^2);
    s = 1/sqrt(1+(alpha/beta)^2);
else
    disp('otherwise');
    c = 1/sqrt(1+(beta/alpha)^2);
    s = -beta/alpha/sqrt(1+(beta/alpha)^2);
end
G = eye(11);
G(4,4) = c;
G(4,11) = s;
G(11,4) = -s;
G(11,11) = c

disp('incrementally applying the Givens rotation to our augmented system')
gAt_aug = G'*At_aug

disp('compare this with the batch QR factorization of At_aug');
[Qt_aug,Rt_aug] = QR(At_aug)

disp('and also compare with the Cholesky factorization of the augmented system');
R_aug_chol = chol(At_aug'*At_aug, 'upper')

more off;