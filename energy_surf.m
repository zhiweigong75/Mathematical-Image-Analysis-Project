%%energy_surf
%
% Author: Zhiwei Gong
% Date: May 10, 2022
%% Plot the energy surface

n = 256;

% Test image 
I = imread('image-house.gif');
I = imresize(I,[n n]);
I = I(:,:,1);

X = im2double(I);
%RI = imref2d(size(X));


G = fft2(I);

% magnitude in log(1+|G|)
AG1 = log(1+abs(G));

SAG1 = fftshift(AG1);

X = 0:16:(n-1);

figure

surf(X,X,AG1(X+1,X+1),'FaceAlpha',0.5)
title('Magnitude of DFT','interpreter','latex')
set(gca,'FontSize', 14);
zlim([0,15])
colorbar;
