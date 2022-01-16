classdef ekf < handle
    % Extended Kalman filter class for state estimation of a nonlinear system
    %
    %   Author: Maani Ghaffari Jadidi
    %   Date:   11/25/2018
    
    properties
        A;              % system matrix Jacobian
        B;              % input matrix Jacobian
        H;              % measurement model Jacobian
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
        
    end
    
    methods
        function obj = ekf(system, init)
            % ekf Construct an instance of this class
            %
            %   Inputs:
            %       system          - system and noise models
            %       init            - initial state mean and covariance
            
            obj.A = system.A;
            obj.B = system.B;
            obj.f = system.f;
            obj.H = system.H;
            obj.Q = system.Q;
            obj.R = system.R;
            obj.h = system.h;
            obj.x = init.x;
            obj.Sigma = init.Sigma;
        end
        
        function prediction(obj)
            % EKF propagation (prediction) step
            obj.x_pred = obj.f(obj.x);
            obj.Sigma_pred = obj.A * obj.Sigma * obj.A' + obj.Q;
            obj.z_hat = obj.h(obj.x_pred);
        end
        
        function correction(obj, z)
            % EKF correction step
            %
            %   Inputs:
            %       z          - measurement
            
            % evaluate measurement Jacobian at current operating point
            H = obj.H(obj.x_pred);
            
            % compute innovation statistics
            % We know here z(2) is an angle
            obj.v = z - obj.z_hat;
            obj.v(2) = wrapToPi(obj.v(2));
            obj.S = H * obj.Sigma_pred * H' + obj.R;
            
            % filter gain
            obj.K = obj.Sigma_pred * H' * (obj.S \ eye(size(obj.S)));
            
            % correct the predicted state statistics
            obj.x = obj.x_pred + obj.K * obj.v;
            I = eye(length(obj.x));
            obj.Sigma = ...
                (I - obj.K * H) * obj.Sigma_pred * (I - obj.K * H)' ...
                    + obj.K * obj.R * obj.K'; % Joseph update form
        end
    end
end