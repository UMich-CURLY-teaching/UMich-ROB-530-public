function ypr = R2ypr(R)
% Extracts angles from a rotation matrix.
% yaw (Z) pitch (Y) roll (X) angles from rotation matrix.
% Extrinsic rotations; The XYZ system rotates, while xyz is fixed.
% Verified using MATLAB eul = rotm2eul(rotm, 'ZYX')

%{  
    Copyright (C) 2017  Maani Ghaffari Jadidi
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details. 
%}

yaw = atan2(R(2,1),R(1,1));
pitch = atan2(-R(3,1),R(1,1)*cos(yaw)+R(2,1)*sin(yaw));
roll = atan2(R(3,2),R(3,3));

ypr = [yaw, pitch, roll];
