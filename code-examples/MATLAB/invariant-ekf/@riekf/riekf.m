classdef riekf < handle
    % Right-Invariant Extended Kalman filter class for 2D Localization, SE(2).
    %
    %   Author: Maani Ghaffari Jadidi
    %   Date:   01/23/2020
    
    properties
        A;              % error dynamics matrix
        H;              % measurement error matrix
        f;              % process model
        X;              % state vector
        P;              % state covariance
        Q;              % input noise covariance
        N;              % measurement noise covariance
    end
    
    methods
        function obj = riekf(system)
            % riekf Construct an instance of this class
            %
            %   Inputs:
            %       system          - system and noise models
            obj.A = system.A;
            obj.f = system.f;
            obj.H = system.H;
            obj.Q = system.Q;
            obj.N = system.N;
            obj.X = eye(3);
            obj.P = 0.1*eye(3);
        end
        
        function AdX = Ad(obj, X)
           % Adjoint
           AdX = [X(1:2,1:2), [X(2,3); -X(1,3)]; 0 0 1];
        end
        
        function xhat = wedge(obj,x)
            % wedge operation for se(2) to put an R^3 vector into the Lie
            % algebra basis.
            G1 = [0    -1     0
                1     0     0
                0     0     0]; % omega
            G2 = [0     0     1
                0     0     0
                0     0     0]; % v_1
            G3 = [0     0     0
                0     0     1
                0     0     0]; % v_2
            xhat = G1 * x(1) + G2 * x(2) + G3 * x(3);
        end
        
        function prediction(obj, u)
            % EKF propagation (prediction) step
            obj.P = obj.A * obj.P * obj.A' + ...
                obj.Ad(obj.X) * obj.Q * obj.Ad(obj.X)';
            obj.X = obj.f(obj.X,u);
        end
        
        function correction(obj, Y1, b1, Y2, b2)
            % RI-EKF correction step
            H = [obj.H(b1); obj.H(b2)]; % stack H
            H = H([1:2,4:5],:); % 4x3 matrix, remove zero rows 
            N = obj.X * blkdiag(obj.N,0) * obj.X'; 
            N = blkdiag(N(1:2,1:2), N(1:2,1:2)); % 4x4 block-diagonal matrix
            % filter gain
            S = H * obj.P * H' + N;
            L = (obj.P * H') * (S \ eye(size(S)));
            
            % Update State
            nu = (blkdiag(obj.X, obj.X) * [Y1; Y2] - [b1; b2]); 
            nu([3,6]) = [];
            delta = obj.wedge( L * nu); % innovation in the spatial frame
            obj.X = expm(delta) * obj.X;
            
            % Update Covariance
            I = eye(size(obj.P));
            obj.P = (I - L * H) * obj.P * (I - L * H)' + L * N * L'; 
        end
    end
end
