classdef ukf < handle & unscented_transform
    % Unscented Kalman filter class for state estimation of a nonlinear system
    %
    %   Author: Maani Ghaffari Jadidi
    %   Date:   01/17/2019
    
    properties
        S;              % innovation covariance
        v;              % innovation
        f;              % process model
        h;              % measurement model
        x;              % state vector
        z_hat;          % predicted measurement
        Sigma;          % state covariance
        x_pred;         % predicted state
        Sigma_pred;     % predicted state covariance
        Q;              % input noise covariance
        R;              % measurement noise covariance
        K;              % Kalman (filter) gain
        Cov_xz;         % state-measurement cross covariance
        kappa_f;        % user-defined parameter to control the sigma points
        kappa_h;        % user-defined parameter to control the sigma points
        ut;             % UT instance for prediction and correction
    end
    
    methods
        function obj = ukf(system, init)
            % ukf construct an instance of this class
            %
            %   Inputs:
            %       system          - system and noise models
            %       init            - initial state mean and covariance
            
            obj.f = system.f;
            obj.Q = system.Q;
            obj.R = system.R;
            obj.h = system.h;
            obj.x = init.x;
            obj.Sigma = init.Sigma;
            obj.ut = unscented_transform();
            obj.kappa_f = init.kappa_f;
            obj.kappa_h = init.kappa_h;
        end
        
        function prediction(obj)
            % UKF propagation (prediction) step
            obj.ut.set(obj.x, obj.Sigma, obj.f, obj.kappa_f);
            obj.ut.propagate;
            obj.x_pred = obj.ut.mean;
            obj.Sigma_pred = obj.ut.Cov + obj.Q;
            % compute new sigma points to predict the measurement
            obj.ut.set(obj.x_pred, obj.Sigma_pred, obj.h, obj.kappa_h);
            obj.ut.propagate;
            obj.z_hat = obj.ut.mean;
        end
        
        function correction(obj, z)
            % UKF correction step
            %
            %   Inputs:
            %       z          - measurement
                        
            % compute innovation statistics
            % We know here z(2) is an angle
            obj.v = z - obj.z_hat;
            obj.v(2) = wrapToPi(obj.v(2));
            obj.S = obj.ut.Cov + obj.R;
            
            % compute state-measurement cross covariance
            obj.Cov_xz = obj.ut.Cov_xy;
            
            % filter gain
            obj.K = obj.Cov_xz * (obj.S \ eye(size(obj.S)));
            
            % correct the predicted state statistics
            obj.x = obj.x_pred + obj.K * obj.v;
            obj.Sigma = obj.Sigma_pred - obj.K * obj.S * obj.K';
        end
    end
end