function X_fista = prox(y_para,lambda_para,mode)
dim = length(y_para);
Y = y_para;
D = eye(dim);
opts.pos = mode;
opts.lambda = lambda_para;
opts.backtracking = false;
X_fista = fista_lasso(Y,D,[],opts);

end