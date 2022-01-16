classdef unscented_transform < handle
    % Unscented transform class for uncertinty propagation through a
    % nonlinear model. For the algorithm, see State Estimation for Robotics, 
    % Timothy D. Barfoot, 2018, Chapter 4.
    %
    %   Author: Maani Ghaffari Jadidi
    %   Date:   01/17/2019
    
    properties
        x_input;        % input mean
        P;              % input covariance
        L;              % scaled Cholesky factor of P
        func;           % nonlinear model
        mean;           % output mean
        Cov;            % output covariance
        Cov_xy;         % input-output cross covariance
        kappa;          % user-defined parameter to control the sigma points
        n;              % input dimention
        X;              % 2n+1 sigma points
        Y;              % mapped sigma points
        w;              % 2n+1 sigma points weights
        
    end
    
    methods
        function obj = unscented_transform(mean, cov, f, kappa)
            % unscented_transform construct an instance of this class
            if nargin == 0
                obj.x_input = [];
                obj.P = [];
                obj.func = [];
                obj.n = [];
                obj.kappa = [];
            elseif nargin == 4
                obj.x_input = mean;
                obj.P = cov;
                obj.func = f;
                obj.n = numel(mean);
                obj.kappa = kappa;
            else
                error('Input must be mean, cov, function, and kappa')
            end
        end
        
        function set(obj, mean, cov, f, kappa)
            % set the input after the instance constructed and used
            obj.x_input = mean;
            obj.P = cov;
            obj.func = f;
            obj.n = numel(mean);
            obj.kappa = kappa;
        end
            
        function sigma_points(obj)
            % sigma points around the reference point
            obj.L = sqrt(obj.n + obj.kappa) * chol(obj.P, 'lower');
            Y = obj.x_input(:, ones(1, numel(obj.x_input)));
            obj.X = [obj.x_input, Y + obj.L, Y - obj.L];
            obj.w = zeros(2 * obj.n + 1, 1);
            obj.w(1) = obj.kappa / (obj.n + obj.kappa);
            obj.w(2:end) = 1 / (2*(obj.n + obj.kappa));
        end
        
        function propagate(obj)
            % propagate the input Gaussian using an unscented transform
            obj.sigma_points();
            % compute sample mean and covariance
            obj.mean = 0;
            obj.Cov = 0;
            obj.Y = [];
            for i = 1:2*obj.n+1
                obj.Y(:,i) = obj.func(obj.X(:,i));
                obj.mean = obj.mean + obj.w(i) * obj.Y(:,i);
            end
            obj.Cov = (obj.Y - obj.mean) * diag(obj.w) * (obj.Y - obj.mean)';
            obj.Cov_xy = (obj.X - obj.x_input) * diag(obj.w) * (obj.Y - obj.mean)';
        end
    end
end

