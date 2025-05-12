function [objCMC,alpha,beta,phi,objFun] = PLSC(data,para,alpha0,beta0,phi0,mode)
% --------------------------------------------------------------------
% Structured sparse partial least square coherence (ssPLSC) algorithm
% handle fused lasso and brain connectivity 
% --------------------------------------------------------------------
% Input:
%       - data.X, EEG complex data matrix, n (sample) * p (EEGsensors)
%       - data.Y, EMG complex data matrix, n (sample) * q (EEGsensors)
%       - data.PX, 
%       - data.PY, 
%       - data.Sxx, 
%       - data.Syy, 
%       - data.Sxy, 
%       - paras: lambda1, lambda2, gamma1, gamma2
%       - alpha0, beta0, phi0: initial value
%       - mode: Non-Negativity Constraint (true/false)
%
% Output:
%       - alpha, weight of X
%       - beta, weight of Y
%       - phi, artificial variable
%       - objFun, value of objective function
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

% Structured penalty
if ~isfield(data,'PX')
    PX = get_connectivity(X);
else
    PX = data.PX;
end

if ~isfield(data,'PY')
    PY = get_connectivity(Y);
else
    PY = data.PY;
end

% Normalization
X = zscore(X);
Y = zscore(Y);

% Set parameters
lambda1 = para(1); lambda2 = para(2);
gamma1 = para(3); gamma2 = para(4);

% Initialization
alpha_new = alpha0;
beta_new = beta0;
phi_new = phi0;

if ~isfield(data,'Sxx')
    Sxx = X'*X;
else
    Sxx = data.Sxx;
end

if ~isfield(data,'Syy')
    Syy = Y'*Y;
else
    Syy = data.Syy;
end

if ~isfield(data,'Sxy')
    Sxy = X'*Y;
else
    Sxy = data.Sxy;
end

H_alpha = eye(EEGsensors)+gamma1*PX;
H_beta = eye(EMGsensors)+gamma2*PY;
Tx = 1/max(eig(H_alpha));
Ty = 1/max(eig(H_beta));

opts.Sxy = Sxy;
opts.H_alpha = H_alpha;
opts.H_beta = H_beta;
opts.lambda1 = lambda1;
opts.lambda2 = lambda2;
LM_para = 1;

max_iter = 200;
objFun = zeros(1,max_iter);
objCMC = zeros(1,max_iter);
err = 1e-9; % 0.01 ~ 0.001
diff_obj = err*10;

% 
for iter = 1:max_iter
    
    Sxy_phi = real(Sxy*exp(-phi_new*1i));
    % Solved alpha, fixed beta and phi (fista)
    y_para = alpha_new+Tx*(Sxy_phi*beta_new-H_alpha*alpha_new);
    lambda_para = lambda1*Tx;
    alpha_new = prox(y_para,lambda_para,mode);

    % update alpha
    if norm(alpha_new) ~= 0
        alpha_new = alpha_new/sqrt(alpha_new'*H_alpha*alpha_new);
    end

    % Solved beta, fixed alpha and phi (fista)
    y_para = beta_new+Ty*(Sxy_phi'*alpha_new-H_beta*beta_new);
    lambda_para = lambda2*Ty;
    beta_new = prox(y_para,lambda_para,mode);

    % update beta
    if norm(beta_new) ~= 0
        beta_new = beta_new/sqrt(beta_new'*H_beta*beta_new);
    end

    % Solved phi, fixed alpha and beta (LM)
    J = Jacobi(alpha_new,beta_new,phi_new,opts);
    gk = J;

    % update phi
    phi0 = phi_new;
    phi_new = phi_new-(J'*J + LM_para)\gk;
    if optFunc(alpha_new,beta_new,phi0,opts) < optFunc(alpha_new,beta_new,phi_new,opts)
        phi_new = phi0;
        LM_para = 10*LM_para;
    end
    
    % Calculate CMC
    
    objCMC(iter) = abs(alpha_new'*Sxy*beta_new)^2/abs((alpha_new'*Sxx*alpha_new)*(beta_new'*Syy*beta_new));
    if isnan(objCMC(iter))
        objCMC(iter) = 0;
    end

    % Cost function and check convergence
    
    objFun(iter) = optFunc(alpha_new,beta_new,phi_new,opts);
    
    if iter ~= 1
        diff_obj = abs((objFun(iter)-objFun(iter-1))/objFun(iter-1)); % relative prediction error
        % plot(iter,diff_obj,'o')
    end

    if diff_obj < err
        % hold off;
        break;
    end

end

% update parameter
alpha = alpha_new;
beta = beta_new;
phi = phi_new;

objFun = objFun(objFun ~= 0);
objCMC = objCMC(objCMC ~= 0);

if isempty(objFun)
    objFun = 0;
end

if isempty(objCMC)
    objCMC = 0;
else
    objCMC = objCMC(end);
end


end


