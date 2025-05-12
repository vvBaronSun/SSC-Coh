function [Fixed_lambda1,Fixed_lambda2] = ParaOpt_sPLSC(data,lambdaSet,TrialNum,EEGsensors,EMGsensors)

alpha0 = ones(EEGsensors,1)/EEGsensors;
beta0 = ones(EMGsensors,1)/EMGsensors;
phi0 = pi/4;
obj_Para = zeros(length(lambdaSet),length(lambdaSet));
indices = crossvalind('KFold',TrialNum,5);
for k = 1:5
    trainFlag = find(indices ~= k); testFlag = find(indices == k);
    trainData.X = data.X(trainFlag,:); trainData.Y = data.Y(trainFlag,:);
    trainData.PX = get_connectivity(trainData.X);
    trainData.PY = get_connectivity(trainData.Y);
    trainData.Sxx = zscore(trainData.X)'*zscore(trainData.X);
    trainData.Sxy = zscore(trainData.X)'*zscore(trainData.Y);
    trainData.Syy = zscore(trainData.Y)'*zscore(trainData.Y);
    testData.X = data.X(testFlag,:); testData.Y = data.Y(testFlag,:);
    Sxy = testData.X'*testData.Y;
    Sxx = testData.X'*testData.X;
    Syy = testData.Y'*testData.Y;
    for lambda1 = 1:length(lambdaSet)
        for lambda2 = 1:length(lambdaSet)
            para = [lambdaSet(lambda1) lambdaSet(lambda2) 0 0];
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