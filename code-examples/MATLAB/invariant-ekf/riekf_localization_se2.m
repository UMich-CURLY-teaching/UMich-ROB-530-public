% Right-Invariant EKF localization on SE(2). The process model 
% is simply X_k+1 = X_k * exp(u_k + w_k) where
% X_k is in SE(2), u_k is the twist in se(2), and w_k is N(0,Q_k) and defined 
% in the Lie algebra se(2). The measurements are noisy 2D coordinates of the
% landmarks in Cartesian plane. We use expm and logm as numerical Lie exp 
% and log map. Both maps have closed-form formulas as well.
%
%   Author: Maani Ghaffari Jadidi
%   Date:   03/12/2020

clc; clear; close all

% generate a path
l = 30; % scale the simulation
dt = 0.2;
gt = [];
gt.x = [0:dt:l, l * ones(1,length(0:dt:l)), l:-dt:0, ...
    zeros(1,length(l:-dt:0))];
gt.y = [0 * ones(1,length(0:dt:l)), 0:dt:l, ...
    l * ones(1,length(l:-dt:0)), l:-dt:0,];

% find the headings tangent to the path
gt.h = 0;
for i = 2:length(gt.x)
   gt.h(i) = atan2(gt.y(i) - gt.y(i-1), gt.x(i) - gt.x(i-1));
end
% ground truth poses
H = cell(length(gt.x),1);
H{1} = eye(3);
for i = 2:length(gt.x)
    H{i} = posemat(gt.x(i), gt.y(i), gt.h(i));
end
% generate noise-free twist control inputs (velocity commands) in the Lie algebra
u = cell(length(gt.x),1);
u{1} = zeros(3);
for i = 2:length(gt.x)
    u{i} = logm(H{i-1} \ H{i});
end

% construct noise free motion trajectory (sanity check for the generated
% inputs!)
path = [];
path.T = H{1};
path.x = 0;
path.y = 0;
for i = 2:size(u,1)
    path.T = path.T * expm(u{i});
    path.x(i+1) = path.T(1,3);
    path.y(i+1) = path.T(2,3);
end

% map of landmarks
landmarks = l * [0.2, 0.2
       0.5, 0.3
       0.8, 0.2
       0.7, 0.5
       0.8, 0.8
       0.5, 0.7
       0.2, 0.8
       0.3, 0.5];
Map = KDTreeSearcher(landmarks);


% build a system 
sys = [];
% motion model noise covariance
sys.Q = diag([0.015^2, 0.01^2, 0.01^2]);
sys.A = eye(3);
sys.f = @(x,u) x*expm(u);
sys.H = @(m) [m(2) -1 0; -m(1) 0 -1; 0 0 0];
sys.N = diag([0.5^2; 0.5^2]);

% se(2) Lie algebra basis twist = vec(\omega, v_1, v_2)
G1 = [0    -1     0
    1     0     0
    0     0     0];
G2 = [0     0     1
    0     0     0
    0     0     0];
G3 = [0     0     0
    0     0     1
    0     0     0];
% now make the twist noisy! in practice the velocity readings are not
% perfect.
% Cholesky factor of covariance for sampling
LQ = chol(sys.Q, 'lower');
for i = 1:length(u)
    noise = LQ * randn(3,1);
    N = G1 * noise(1) + G2 * noise(2) + G3 * noise(3);
    u{i} = u{i} + N;
end

% incremental visualization
green = [0.2980 .6 0];
crimson = [220,20,60]/255; 
darkblue = [0 .2 .4];
Darkgrey = [.25 .25 .25];
VermillionRed = [156,31,46]/255;
DupontGray = [144,131,118]/255;

fsize = 14; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');

h_leg = []; % legend handle
figure; hold on
h_leg{1} = plot(path.x, path.y, '-', 'color', Darkgrey, 'linewidth', 3);
grid on, axis auto equal, axis([-6 6 -1 5])
xlabel('$x_1$', 'fontsize', fsize, 'Interpreter','latex')
ylabel('$x_2$', 'fontsize', fsize, 'Interpreter','latex')
set(gca, 'fontsize', fsize)
axis equal, axis([-l/4 l*1.35 -l/4 l*1.35]), axis off

% plot map
plot(Map.X(:,1), Map.X(:,2), '*k', 'MarkerSize', 10)
plot(Map.X(:,1), Map.X(:,2), 'sk', 'MarkerSize', 10)

% Cholesky factor of measurement noise covariance for sampling
LN = chol(sys.N, 'lower');

filter = riekf(sys); % create an RI-EKF object

% plot initial mean 
h_leg{2} = plot(filter.X(1,3), filter.X(2,3), 'o', 'color', [crimson, 0.7], 'MarkerFaceColor', crimson, 'markersize', 8);
h_leg{3} = quiver(filter.X(1,3), filter.X(2,3), 10*filter.X(1,1), 10*filter.X(2,1), 'LineWidth', 2, 'color', darkblue);
ELLIPSE = confidence_ellipse(filter.X, chol(filter.P, 'lower'));
h_leg{4} = plot(ELLIPSE(:,1), ELLIPSE(:,2), 'color', [VermillionRed, 0.7], 'linewidth', 2);
h_leg{5} = line([filter.X(1,3), Map.X(1,1)], [filter.X(2,3), Map.X(2,1)],'Color',[green, 0],'LineStyle','-', 'linewidth', 2);

% video recorder object
% video = VideoWriter('dead-reckoning_se2.avi', 'Motion JPEG AVI');
video = VideoWriter('riekf_localization_se2.avi', 'Motion JPEG AVI');
video.Quality = 100;
open(video)

skip = 50;
for i = 1:size(u,1)
    % predict next pose using given twist
    filter.prediction(u{i});
    
    if ~mod(i,skip)
        % get a landmark measurement using current true position of the robot
        m_id = knnsearch(Map, [gt.x(i), gt.y(i)], 'K', 2);
        b1 = [Map.X(m_id(1),:), 1]';
        Y1 = H{i} \ b1 + [LN * randn(2,1); 0];
        b2 = [Map.X(m_id(2),:), 1]';
        Y2 = H{i} \ b2 + [LN * randn(2,1); 0];
        % correction based on the measurements
        filter.correction(Y1, b1, Y2, b2);
    end
    
    % update graphics
    set(h_leg{2},'XData',filter.X(1,3),'YData', filter.X(2,3));
    set(h_leg{3},'XData',filter.X(1,3),'YData', filter.X(2,3), 'UData', filter.X(1,1), 'VData', filter.X(2,1));
    ELLIPSE = confidence_ellipse(filter.X, chol(filter.P, 'lower'));
    set(h_leg{4},'XData',ELLIPSE(:,1),'YData', ELLIPSE(:,2));
    if ~mod(i,skip)
        for j = 1:length(m_id)
            set(h_leg{5},'XData',[filter.X(1,3), Map.X(m_id(j),1)],'YData', [filter.X(2,3), Map.X(m_id(j),2)], 'Color',[green, 0.5]);
            drawnow limitrate
%             pause(0.01)
        end
    else
        drawnow limitrate
%         pause(0.01)
    end
    drawnow limitrate
    frame = getframe(gcf);
    writeVideo(video,frame);
end

close(video)


% figuresize(21,21,'cm')
% print -opengl -dpng -r600 riekf_loc_se2.png


function H = posemat(x,y,h)
% construct a SE(2) matrix element
H = [cos(h) -sin(h) x;
    sin(h)  cos(h)  y;
    0       0       1];
end

function ELLIPSE = confidence_ellipse(X,L)
% create confidence ellipse
% se(2) Lie algebra basis twist = vec(\omega, v_1, v_2)
G1 = [0    -1     0
      1     0     0
      0     0     0];
G2 = [0     0     1
      0     0     0
      0     0     0];
G3 = [0     0     0
      0     0     1
      0     0     0];

% first create points from a unit circle + angle (third dimension of so(3))
phi = (-pi:.01:pi)';
circle = [zeros(length(phi),1), cos(phi), sin(phi)];
% Chi-squared 3-DOF 95% percent confidence (0.05): 7.815
scale = sqrt(7.815);
% main loop; iterate over the control inputs and move the robot
ELLIPSE = zeros(size(circle,1),2); % covariance ellipse on manifold (nonlinear)

for j = 1:size(circle,1)
    % sample covariance on SE(2)
    ell_se2_vec = scale * L * circle(j,:)';
    % retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
    temp = X * expm(G1 * ell_se2_vec(1) + G2 * ell_se2_vec(2) + G3 * ell_se2_vec(3));
    ELLIPSE(j,:) = [temp(1,3), temp(2,3)];
end

end