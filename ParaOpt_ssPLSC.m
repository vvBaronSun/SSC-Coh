function [Fixed_lambda1,Fixed_lambda2,Fixed_gamma1,Fixed_gamma2] = ParaOpt_ssPLSC(data,lambdaSet,gammaSet,TrialNum,EEGsensors,EMGsensors)

alpha0 = ones(EEGsensors,1)/EEGsensors;
beta0 = ones(EMGsensors,1)/EMGsensors;
phi0 = pi/4;

obj_Para = zeros(length(gammaSet),length(gammaSet));
indices = crossvalind('KFold',TrialNum,5);
for k = 1:5
    trainFlag = find(indices ~= k); testFlag = find(indices == k);
    trainData.X = data.X(trainFlag,:); trainData.Y = data.Y(trainFlag,:);
    testData.X = data.X(testFlag,:); testData.Y = data.Y(testFlag,:);
    trainData.PX = get_connectivity(trainData.X);
    trainData.PY = get_connectivity(trainData.Y);
    trainData.Sxx = zscore(trainData.X)'*zscore(trainData.X);
    trainData.Sxy = zscore(trainData.X)'*zscore(trainData.Y);
    trainData.Syy = zscore(trainData.Y)'*zscore(trainData.Y);
    Sxy = testData.X'*testData.Y;
    Sxx = testData.X'*testData.X;
    Syy = testData.Y'*testData.Y;
    for gamma1 = 1:length(gammaSet)
        for gamma2 = 1:length(gammaSet)
            para = [0 0 gammaSet(gamma1) gammaSet(gamma2)];
            [obj_SCCMC,alpha,beta] = PLSC(trainData,para,alpha0,beta0,phi0,false);
            if sum(alpha) ~= 0 && sum(beta) ~= 0
                est_SCCMC = abs(alpha'*Sxy*beta)^2/abs((alpha'*Sxx*alpha)*(beta'*Syy*beta));
            else
                est_SCCMC = 10;
            end
            obj_Para(gamma1,gamma2) = obj_Para(gamma1,gamma2)+...
                abs(est_SCCMC-obj_SCCMC(end))/obj_SCCMC(end);
        end
    end
end
[row,col] = find(obj_Para == min(obj_Para,[],'all'));
Fixed_gamma1 = gammaSet(row(1));
Fixed_gamma2 = gammaSet(col(1));


obj_Para = zeros(length(lambdaSet),length(lambdaSet));
indices = crossvalind('KFold',TrialNum,5);
for k = 1:5
    trainFlag = find(indices ~= k); testFlag = find(indices == k);
    trainData.X = data.X(trainFlag,:); trainData.Y = data.Y(trainFlag,:);
    testData.X = data.X(testFlag,:); testData.Y = data.Y(testFlag,:);
    trainData.PX = get_connectivity(trainData.X);
    trainData.PY = get_connectivity(trainData.Y);
    trainData.Sxx = zscore(trainData.X)'*zscore(trainData.X);
    trainData.Sxy = zscore(trainData.X)'*zscore(trainData.Y);
    trainData.Syy = zscore(trainData.Y)'*zscore(trainData.Y);
    Sxy = testData.X'*testData.Y;
    Sxx = testData.X'*testData.X;
    Syy = testData.Y'*testData.Y;
    for lambda1 = 1:length(lambdaSet)
        for lambda2 = 1:length(lambdaSet)
            para = [lambdaSet(lambda1) lambdaSet(lambda2) Fixed_gamma1 Fixed_gamma2];
            [obj_SCCMC,alpha,beta] = PLSC(trainData,para,alpha0,beta0,phi0,false);
            if sum(alpha) ~= 0 && sum(beta) ~= 0
                est_SCCMC = abs(alpha'*Sxy*beta)^2/abs((alpha'*Sxx*alpha)*(beta'*Syy*beta));
            else
                est_SCCMC = 10;
            end
            obj_Para(lambda1,lambda2) = obj_Para(lambda1,lambda2)+...
                abs(est_SCCMC-obj_SCCMC(end))/obj_SCCMC(end);
        end
    end
end
[row,col] = find(obj_Para == min(obj_Para,[],'all'));
Fixed_lambda1 = lambdaSet(row(1));
Fixed_lambda2 = lambdaSet(col(1));

end