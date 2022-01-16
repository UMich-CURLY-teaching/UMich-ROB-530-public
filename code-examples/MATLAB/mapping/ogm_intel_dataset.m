clc; clear; close all

load('sample_Intel_dataset.mat');

green = [0.2980 .6 0];
crimson = [220,20,60]/255; 
darkgrey = [.35 .35 .35];

ogm = occupancy_grid_map(robotPose, laserScan);
ogm.build_ogm;
ogm.plot;

fsize = 20; % font size
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
% plot(robotPose.x, robotPose.y, 'linewidth', 2)
set(gca, 'fontsize', fsize)
figuresize(21,21,'cm')
% print -opengl -dpng -r600 ogm_intel.png

% plot the point cloud map and the robot trajectory 
figure; hold on;
plot(robotPose.x, robotPose.y, 'linewidth', 2)
for i = 1:length(laserScan)
    plot(globalLaserScan{i}(1,:), globalLaserScan{i}(2,:),'.', 'color', [darkgrey, .1]); 
end
plot(robotPose.x(1), robotPose.y(1), 's', 'color', [green, 0.7], 'MarkerFaceColor', green, 'markersize', 14)
plot(robotPose.x(end), robotPose.y(end), 'o', 'color', [crimson, 0.7], 'MarkerFaceColor', crimson, 'markersize', 14)
axis equal tight
set(gca, 'fontsize', fsize)
figuresize(21,21,'cm')
% print -opengl -dpng -r600 ptcloud_intel.png