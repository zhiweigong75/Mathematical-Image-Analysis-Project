%%script_deconvolution_2d
%
% This script is used for deconvolution on Time domain
%
% Author: Jan Glaubitz
% Modifier: Zhiwei Gong
% Date: May 10, 2022
%
clearvars; close all; clc; % clean up
%% Free parameters 

% Free parameters
n = 256; % number of pixels in every direction 
%ind = 0:20:399;
gamma = 0.01; % blurring parameter (Gaussian convolution kernel) 
noise_variance = 10^(-3); % variance of the i.i.d. complex Gaussian noise added to the measurements 
order = 2; % order of the TV/PA operator (1,2,3) 
c = 1; d = 0.001; % hyper-hyper-parameters


%% Set up the model 

% Test image 
I = imread('image-polygons.gif');
I = I(:,:,1);%

X = im2double(I);
%X = imresize(I,[n n]);
RI = imref2d(size(X)); %
x = X(:); % vectorize the image by stacking up the columns 

% forward operator, noise, and data 
F_1d = construct_F_deconvolution( n, gamma ); % 1d forward operator 
rng('default'); rng(1,'twister'); % to make the results reproducable 
noise = sqrt(noise_variance/2)*randn(size(F_1d,1),size(F_1d,1)); % iid real Gaussian noise 
Y = F_1d*X*(F_1d') + noise; % real-valued noisy indirect measuremnt 
y = Y(:); 

% Regularization operator  
D = TV_operator( n, order ); 



%% Use different methods for reconstruction 

% SBL based on Bayesian coordinate descent 
[Mu, alpha, B1, B2, history] = BCD_2d( F_1d, Y, D, c, d );
X_BCD = Mu;  

% Compute SNR 
SNR = norm(x)^2/(length(x)*noise_variance);


%% Plot the results 

% Exact image 
f = figure(1);

subplot(1,3,1)

%ax = axes(f);
imshow(X, RI, 'InitialMagnification',600); 
%colorbar;
title('Exact image')
set(gca, 'FontSize', 16); % Increasing ticks fontsize 


% Noisy blurred image 
%f = figure(2);
subplot(1,3,2)
%ax = axes(f);
imshow(Y, RI, 'InitialMagnification',600); 
%colorbar;
title('Noisy blurred image')
set(gca, 'FontSize', 16); % Increasing ticks fontsize 

% SBL by BCD 
%f = figure(4);
subplot(1,3,3)
%ax = axes(f);
imshow(X_BCD, RI, 'InitialMagnification',600); 
%colorbar;
title('SBL by BCD')
set(gca, 'FontSize', 16); % Increasing ticks fontsize 
