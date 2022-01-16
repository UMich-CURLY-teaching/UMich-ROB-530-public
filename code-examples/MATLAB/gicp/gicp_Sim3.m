function T = gicp_Sim3(target, source, varargin)
% Generalized ICP over Sim(3)
%
%   Author: Maani Ghaffari Jadidi
%   Date:   03/11/2020

% Create kd-tree objects for NN queries
target_xyz = squeeze(double(target.Location));
source_xyz = squeeze(double(source.Location));
target_kdt = KDTreeSearcher(target_xyz);
source_kdt = KDTreeSearcher(source_xyz);

% Covariance normal at each point 
Ct = pc_covariances_ann(target_kdt);
Cs = pc_covariances_ann(source_kdt);

% Initial guess
T0 = eye(4);
T1 = T0;

% ICP loop: find correspondences and optimize4
d_threshold = 1.5;
converged = false;
tf_epsilon = 1e-5;
iter = 0;
max_iter = 50;
inner_max_iter = 100;
eps_Jr = 1e-6;
while ~converged && iter < max_iter
    % apply the current transformation to the source point cloud
    current_source = source_xyz * T0(1:3,1:3)';
    current_source(:,1) = current_source(:,1) + T0(1,4);
    current_source(:,2) = current_source(:,2) + T0(2,4);
    current_source(:,3) = current_source(:,3) + T0(3,4);
    
    % NN queries
    idx = knnsearch(target_kdt, current_source);
    
    % apply distance threshold to remove outliers
    dist = sqrt(sum((current_source - target_kdt.X(idx,:)).^2,2));
    survived_idx = find(dist < d_threshold);
    target_idx = idx(dist < d_threshold);
    
    p_source = source_xyz(survived_idx,:);
    p_target = target_kdt.X(target_idx,:);
    
    % solve for the new transformation
    % Gauss-Newton solver over Sim(3)
    inner_iter = 0;
    while inner_iter < inner_max_iter
        inner_iter = inner_iter + 1;
        % solve normal equations
        [A,b] = compute_jacobian(T1);
        dx = A \ b;
        % retract and update the estimate
        T1 = expm( hat(dx) ) * T1;
        
        if ~mod(inner_iter,5)
            disp(['GN Iter: '     num2str(inner_iter)])      
        end
        
        % check if converged
        if norm(b) < eps_Jr
            if mod(inner_iter,5)
            disp(['GN Iter: '     num2str(inner_iter)])      
            end
            break;
        end
    end

    % check if converged
    if norm(logm(T0 \ T1)) < tf_epsilon
        disp('Converged')
        converged = true;
    else
        T0 = T1;
        iter = iter + 1;
        disp(['Iter: '     num2str(iter)])
        if ~(iter < max_iter)
            disp(['Not converged. Maximum iteration of ', num2str(max_iter), ' is reached'])
        end
    end
end

T = T1;
disp('Transformation: ')
disp(T)
disp('Scale: ')
disp(det(T(1:3,1:3)))

% A and b for GN
function [A,b] = compute_jacobian(X)
    A = zeros(7);
    b = zeros(7,1);
    sR = X(1:3,1:3); % scale + rotation
    t = X(1:3,4);
    % residual
    r = p_target - p_source * sR';
    r(:,1) = r(:,1) - t(1);
    r(:,2) = r(:,2) - t(2);
    r(:,3) = r(:,3) - t(3);
    n = size(r,1);
    for i = 1:n
        % Inverse of covariance Cholesky factor
        invL = chol(Ct{target_idx(i)} + sR * Cs{survived_idx(i)} * sR', 'lower') \ eye(3);
        % Jacobian
        J = invL * [skew(p_source(i,:)'), -eye(3), -p_source(i,:)'];
        % Left hand side matrix A x = b
        A = A + J' * J;
        % Right hand side vector A x = b
        b = b - J' * invL * r(i,:)';
    end
end

function C = pc_covariances_ann(pckdt)
% Compute the empirical covariance at each point using an ANN search
    e = 1e-2; % covariance epsilon
    C = cell(size(pckdt.X,1),1);
    for i = 1:length(C)
        nn_id = knnsearch(pckdt, pckdt.X(i,:), 'K', 6);
        % HACK: I'm adding a jitter to avoid singularity.
        Cov = cov(pckdt.X(nn_id,:)) + e * eye(3);
        % GICP covariance
        [V, D] = eig(Cov);
        D(1,1) = e;
        D(2,2) = 1;
        D(3,3) = 1;
        C{i} = V * D * V';
    end
end

function X = skew(x)
% vector to skew R^3 -> so(3)
X = [   0,  -x(3),  x(2);
    x(3),      0,  -x(1);
    -x(2), x(1),   0];
end

function X = hat(x)
% hat: R^7 -> sim(3)
X = [x(7) * eye(3) + skew(x(1:3)), x(4:6); 0 0 0 0];
end

end
