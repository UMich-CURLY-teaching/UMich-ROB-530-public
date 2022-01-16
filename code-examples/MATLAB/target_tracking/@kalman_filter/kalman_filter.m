classdef kalman_filter < handle
    % Kalman filter class for state estimation of a linear system
    %
    %   Author: Maani Ghaffari Jadidi
    %   Date:   11/24/2018
    
    properties
        A;              % system matrix
        B;              % input matrix
        H;              % measurement model
        S;              % innovation covariance
        v;              % innovation
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
        function obj = kalman_filter(system, init)
            % KALMAN_FILTER Construct an instance of this class
            %
            %   Inputs:
            %       system          - linear system and noise models
            %       init            - initial state mean and covariance
            
            obj.A = system.A;
            obj.B = system.B;
            obj.H = system.H;
            obj.Q = system.Q;
            obj.R = system.R;
            obj.x = init.x;
            obj.Sigma = init.Sigma;
        end
        
        function prediction(obj)
            % Kalman filter propagation (prediction) step
            obj.x_pred = obj.A * obj.x; % + obj.B * u;
            obj.Sigma_pred = obj.A * obj.Sigma * obj.A' + obj.Q;
            obj.z_hat = obj.H * obj.x_pred;
        end
        
        function correction(obj, z)
            % Kalman filter correction step
            %
            %   Inputs:
            %       z          - measurement
            
            % compute innovation statistics
            obj.v = z - obj.z_hat;
            obj.S = obj.H * obj.Sigma_pred * obj.H' + obj.R;
            
            % filter gain
            obj.K = obj.Sigma_pred * obj.H' * (obj.S \ eye(size(obj.S)));
            
            % correct the predicted state statistics
            obj.x = obj.x_pred + obj.K * obj.v;
            I = eye(size(obj.x));
            obj.Sigma = ...
                (I - obj.K * obj.H) * obj.Sigma_pred * (I - obj.K * obj.H)' ...
                    + obj.K * obj.R * obj.K'; % Joseph update form
        end
    end
end