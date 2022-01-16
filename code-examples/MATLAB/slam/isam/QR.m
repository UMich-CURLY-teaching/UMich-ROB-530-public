function [Q,R] = QR(A)
%QR factorization
%   [Q,R] = QR(A) where A is a MxN matrix, Q is a MxM orthogonal matrix and
%   R is MxN upper triangular matrix above the diagonal

[m,n] = size(A);

QT = eye(m);
for k=1:n
    for i=k+1:m % i > k
        if A(i,k) == 0
            continue;
        end
        
        % Givens rotation matrix
        alpha = A(k,k);
        beta = A(i,k);
        if beta == 0
            c = 1;
            s = 0;
        elseif abs(beta) > abs(alpha)
            c = -alpha/beta/sqrt(1+(alpha/beta)^2);
            s = 1/sqrt(1+(alpha/beta)^2);
        else
            c = 1/sqrt(1+(beta/alpha)^2);
            s = -beta/alpha/sqrt(1+(beta/alpha)^2);
        end
        G = speye(m);
        G(k,k) = c;
        G(k,i) = s;
        G(i,k) = -s;
        G(i,i) = c;

        QT = G'*QT;
        A = G'*A;
        
        if abs(A(i,k)) < abs(eps)
            A(i,k) = 0;
        end
    end
end

Q = QT';
R = A;