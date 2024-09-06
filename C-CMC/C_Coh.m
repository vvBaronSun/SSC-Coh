function [alpha,beta,phi,objCMC,EEGtopo,EMGtopo] = C_Coh(data,phi0,info)
% --------------------------------------------------------------------
% Canonical coherence (C-Coh) algorithm
% --------------------------------------------------------------------
% Input:
%       - data.X, EEG complex data matrix, n (sample) * p (EEGsensors)
%       - data.Y, EMG complex data matrix, n (sample) * q (EEGsensors)
%       - phi0: initial value
%
% Output:
%       - alpha, weight of X
%       - beta, weight of Y
%       - phi, artificial variable
%       - objCMC, value of optimized corticomuscular coherence
%       - EEGtopo, EEG topography
%       - EMGtopo, EMG topography
%
%---------------------------------------------------------------------
% Author: Jingyao Sun, sunjy22@mails.tsinghua.edu.cn
% Date created: July-30-2024
% @Tsinghua Univertity.
% --------------------------------------------------------------------

% Setting
n_sample = size(data.X,1);
EEGsensors = size(data.X,2);
EMGsensors = size(data.Y,2);

% Set data
X = data.X;
Y = data.Y;

% Normalization
X = zscore(X);
Y = zscore(Y);

% Initialization
Sxx = X'*X;
Syy = Y'*Y;

% PCA decomposition
[newX,U1,EEGdim] = Dimreduction(X,EEGsensors,info);
[newY,U2,EMGdim] = Dimreduction(Y,EMGsensors,info);
newXY = [newX newY];
CMatrix = newXY'*newXY;

% Levenberg- Marquardt algorithm
LPhi_diff0 = LPhidiff(CMatrix,EEGdim,EMGdim,phi0);
options = optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','off');
phi = fsolve(@(x)LPhidiff(CMatrix,EEGdim,EMGdim,x),LPhi_diff0,options);
[objCMC,DPhi,CAA_inv,CBB_inv] = LPhi(CMatrix,EEGdim,EMGdim,phi);


% update parameter
% 
[V1,D1]=eig(DPhi*DPhi');
[V2,D2]=eig(DPhi'*DPhi);
%
[~,ind1] = sort(diag(D1));
Vs1 = V1(:,ind1);
[~,ind2] = sort(diag(D2));
Vs2 = V2(:,ind2);
%
a = Vs1(:,end);
b = Vs2(:,end);

alpha = U1*CAA_inv*a;
beta = U2*CBB_inv*b;

EEGtopo = abs(Sxx)*alpha;
EMGtopo = abs(Syy)*beta;

end


