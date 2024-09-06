function optValue = optFunc(alpha,beta,phi,opts)

Sxy = opts.Sxy;
H_alpha = opts.H_alpha;
H_beta = opts.H_beta;
lambda1 = opts.lambda1;
lambda2 = opts.lambda2;

optValue = -alpha'*real(Sxy*exp(-phi*1i))*beta+lambda1*norm(alpha,1)+lambda2*norm(beta,1)...
        + 0.5*alpha'*H_alpha*alpha + 0.5*beta'*H_beta*beta;

end