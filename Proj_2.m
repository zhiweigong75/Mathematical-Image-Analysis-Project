% Script for deconvolution on Fourier domain
%
% Author: Zhiwei Gong
% Date: May 10, 2022
%
clearvars; close all; clc; % clean up
%% Free parameters 

% Free parameters
n = 256; % number of pixels in every direction 
noise_variance = 10^(-3); % variance of the i.i.d. complex Gaussian noise added to the measurements
order = 2; % order of the TV/PA operator (1,2,3) 
c = 1; d = 0.05; % hyper-hyper-parameters

%% Set up the model 

% Test image 
I = imread('image-polygons.gif');
I = imresize(I,[n n]);
I = I(:,:,1);

X = im2double(I);
RI = imref2d(size(X)); %
x = X(:); % vectorize the image by stacking up the columns



% 
% 
% define the shifted filter
L = zeros(n,n);

for j = 1:n
    
    for k = 1:n
      
        %high-pass
        %L(j,k) = 1-exp(-0.5.*((j-n/2)^2+(k-n/2)^2)/(n/25)^2 ); 
        
        %higher-pass
        L(j,k) = 1-exp(-0.5.*((j-n/2)^2+(k-n/2)^2)/(n/15)^2 );    
        
        % lower-pass
        %L(j,k) = 1/(1+ (sqrt((j-n/2)^2+(k-n/2)^2)/(n/16) )^8  );
        %L(j,k) = exp(-0.5.*((j-n/2)^2+(k-n/2)^2)/(n/25)^2 ); % Gaussian
        
        
        % low-pass
        %L(j,k) = 1/(1+ (sqrt((j-n/2)^2+(k-n/2)^2)/(n/4) )^8  );
        %L(j,k) = exp(-0.5.*((j-n/2)^2+(k-n/2)^2)/(n/15)^2 );  % Gaussian
    end
    
end

L = ifftshift(L);

ind = 0:10:(n-1);

figure(1)
surf(ind,ind,L(ind+1,ind+1),'FaceAlpha',0.5)
title('Gaussian low-pass filter','interpreter','latex')
set(gca,'FontSize', 14);
zlim([0,1])

G = [L, L;L, L];
% Data model and noise 
F_1d_complex = dftmtx(n)/sqrt(n); % matrix corresponding to the one-dimensional normalized DFT 
%F_1d_undersampled = F_1d_complex(samples,:); % only keep a smaller number of rows 
%F_1d_complex(samples_rmvd,:) = [];
F_1d = [real(F_1d_complex); imag(F_1d_complex)]; % real-valued forward operator
rng('default'); rng(1,'twister'); % to make the results reproducable
noise = sqrt(noise_variance/2)*randn(size(F_1d,1),size(F_1d,1)); % iid real Gaussian noise 
Y = G.*(F_1d*X*(F_1d')) + noise; % real-valued noisy indirect measuremnt 
y = Y(:); 

% Regularization operator  
D = TV_operator( n, order ); % regularization operator 




%% Use different methods for reconstruction 
% noisy image

% reconstruct the complex noisy image
Z = Y(1:n,1:n)-Y((n+1):end, (n+1):end) + 1i.*(Y(1:n, (n+1):end)+Y((n+1):end, 1:n));
X_LS = real(ifft2(Z)).*n ;

% normalize

% by F-norm
%X_NLS = X_LS.*(norm(X)./norm(X_LS));

% by l^1-norm
X_NLS = X_LS + mean( X(:) - X_LS(:));


% SBL based on Bayesian coordinate descent 
[Mu, alpha, B1, B2, history] = FBCD_2d( F_1d, G, Y, D, c, d );
X_BCD = Mu;

% normalize
% by F-norm
%X_BCD = X_BCD.*(norm(X)./norm(X_BCD));

% by l^1-norm
X_BCD = X_BCD + mean( X(:) - X_BCD(:));


% Compute SNR 
SNR = norm(x)^2/(length(x)*noise_variance);


%% Plot the results 

% Exact image 
f = figure(2);
%ax = axes(f);
subplot(2,2,1)
imshow(X, RI, 'InitialMagnification',600); 
%colorbar;
title({'Exact'},'interpreter','latex')
set(gca, 'FontSize', 14); % Increasing ticks fontsize 

% Noisy blurred image 
%f = figure(2);
%ax = axes(f);
subplot(2,2,2)
imshow(X_LS, RI, 'InitialMagnification',600); 
title({'Noisy low-pass'},'interpreter','latex')
%colorbar;
set(gca, 'FontSize', 14); % Increasing ticks fontsize 


% Noisy blurred image 
%f = figure(2);
%ax = axes(f);
subplot(2,2,3)
imshow(X_NLS, RI, 'InitialMagnification',600); 
title({'Normalized noisy low-pass'},'interpreter','latex')
%colorbar;
set(gca, 'FontSize', 14); % Increasing ticks fontsize 


% SBL by BCD 
%f = figure(4);
%ax = axes(f);
subplot(2,2,4)
imshow(X_BCD, RI, 'InitialMagnification',600);
title({'SBL by BCD'},'interpreter','latex')
%colorbar;
set(gca, 'FontSize', 14); % Increasing ticks fontsize 




