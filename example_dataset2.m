clear,clc
close all

addpath("SSC-CMC\")
addpath("C-CMC\")
addpath("FISTA-master\")
addpath("Dataset2\")

% Initialize
TrialNum = 30;
PermuteNum = 500; PermutePoint = PermuteNum*0.05;
SF = 1000;

% Parameter set (first optimize gamma, then lambda)
lambdaSet = [0.01 0.1 1 10];
gammaSet = [0.01 0.1 1 10];

% result matrix
subNum = 13; bandNum = 3;
%
CCMC_Sig_Coh = cell(subNum,bandNum);
CCMC_EEGTopo = cell(subNum,bandNum); CCMC_EMGTopo = cell(subNum,bandNum);
CCMC_Coh = cell(subNum,bandNum);
%
SCCMC_Sig_Coh = cell(subNum,bandNum);
SCCMC_EEGTopo = cell(subNum,bandNum); SCCMC_EMGTopo = cell(subNum,bandNum);
SCCMC_Coh = cell(subNum,bandNum);
%
SSCCMC_Sig_Coh = cell(subNum,bandNum);
SSCCMC_EEGTopo = cell(subNum,bandNum); SSCCMC_EMGTopo = cell(subNum,bandNum);
SSCCMC_Coh = cell(subNum,bandNum);
%
f_peak = zeros(subNum,bandNum);
for sub = 1:subNum
    disp("---------Subject"+num2str(sub)+"---------")
    % load data
    filename = strcat(pwd,'\Dataset2\data_',num2str(sub),'.mat');
    load(filename)
    EEGdata = EEGdata(1001:5000,:,1:TrialNum);
    EMGdata = EMGdata(1001:5000,:,1:TrialNum);
    p = size(EEGdata,2); q = size(EMGdata,2);
    % Fourier Transform
    fEEGdata = FourierTransform(EEGdata,SF);
    [fEMGdata,faxis] = FourierTransform(EMGdata,SF);

    % three band
    f_range = [8 13;15 30;30 45];
    ftar_max = zeros(bandNum,1);
    % pre-calculate to find the peak frequency
    alpha0 = ones(p,1)/p;
    beta0 = ones(q,1)/q;
    phi0 = pi/4;
    mode = false;
    para = [0 0 0 0];
    info = 0.95;
    % rand order
    order = zeros(PermuteNum,TrialNum);
    for i = 1:PermuteNum
        order(i,:) = randperm(TrialNum);
    end
    
    %% main
    for band = 1:bandNum
        disp("-----Band"+num2str(band)+"-----")
        f_band = find(faxis > f_range(band,1) & faxis < f_range(band,2));
        CCMC_Sig_Coh{sub,band} = zeros(1,length(f_band));
        CCMC_EEGTopo{sub,band} = zeros(p,length(f_band)); CCMC_EMGTopo{sub,band} = zeros(q,length(f_band));
        SCCMC_Sig_Coh{sub,band} = zeros(1,length(f_band));
        SCCMC_EEGTopo{sub,band} = zeros(p,length(f_band)); SCCMC_EMGTopo{sub,band} = zeros(q,length(f_band));
        SSCCMC_Sig_Coh{sub,band} = zeros(1,length(f_band));
        SSCCMC_EEGTopo{sub,band} = zeros(p,length(f_band)); SSCCMC_EMGTopo{sub,band} = zeros(q,length(f_band));
        for f = 1:length(f_band)
            disp("--faxis"+num2str(f)+"--")
            tic
            data.X = squeeze(fEEGdata(f_band(f),:,:))';
            data.Y = squeeze(fEMGdata(f_band(f),:,:))';
            data.PX = get_connectivity(data.X);
            data.PY = get_connectivity(data.Y);
            data.Sxx = zscore(data.X)'*zscore(data.X);
            data.Sxy = zscore(data.X)'*zscore(data.Y);
            data.Syy = zscore(data.Y)'*zscore(data.Y);
            randData.X = data.X;
            randData.PX = data.PX;
            randData.PY = data.PY;
            randData.Sxx = data.Sxx;
            %% C-CMC algorithm
            % real value
            [~,~,~,obj_CCMC,CCMC_EEGTopo{sub,band}(:,f),CCMC_EMGTopo{sub,band}(:,f)] = C_Coh(data,phi0,info);
            CCMC_Coh{sub,band}(f) = obj_CCMC;
            % permuted value
            rand_CCMC = zeros(1,PermuteNum);
            for r = 1:PermuteNum
                randData.Y = data.Y(order(r,:),:);
                [~,~,~,rand_CCMC(r)] = C_Coh(randData,phi0,info);
            end

            rand_CCMC = sort(rand_CCMC,'descend');
            if obj_CCMC > rand_CCMC(floor(PermutePoint))
                CCMC_Sig_Coh{sub,band}(f) = obj_CCMC-mean(rand_CCMC);
            else
                CCMC_Sig_Coh{sub,band}(f) = 0;
            end
            %% SC-CMC algorithm
            % Parameter optimization
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
                        [alpha,beta,~,~,obj_SCCMC] = SSC_Coh(trainData,para,alpha0,beta0,phi0,mode);
                        if sum(alpha) ~= 0
                            est_SCCMC = abs(alpha'*Sxy*beta)^2/abs((alpha'*Sxx*alpha)*(beta'*Syy*beta));
                        else
                            est_SCCMC = 10;
                        end
                        obj_Para(lambda1,lambda2) = obj_Para(lambda1,lambda2)+...
                            abs(est_SCCMC-obj_SCCMC(end))/obj_SCCMC(end);
                    end
                end
            end
            obj_Para = obj_Para*1000;
            [row,col] = find(obj_Para == min(obj_Para,[],'all'));
            Fixed_lambda1 = lambdaSet(row(1));
            Fixed_lambda2 = lambdaSet(col(1));
            % real value
            para = [Fixed_lambda1 Fixed_lambda2 0 0];
            [~,~,~,~,obj_SCCMC,SCCMC_EEGTopo{sub,band}(:,f),SCCMC_EMGTopo{sub,band}(:,f)] = SSC_Coh(data,para,alpha0,beta0,phi0,mode);
            obj_SCCMC = obj_SCCMC(end);
            SCCMC_Coh{sub,band}(f) = obj_SCCMC;
            % permuted value
            rand_SCCMC = zeros(1,PermuteNum);
            for r = 1:PermuteNum
                randData.Y = data.Y(order(r,:),:);
                [~,~,~,~,SCCMCbuffer] = SSC_Coh(randData,para,alpha0,beta0,phi0,mode);
                rand_SCCMC(r) = SCCMCbuffer(end);
            end

            rand_SCCMC = sort(rand_SCCMC,'descend');
            if obj_SCCMC > rand_SCCMC(floor(PermutePoint))
                SCCMC_Sig_Coh{sub,band}(f) = obj_SCCMC-mean(rand_SCCMC);
            else
                SCCMC_Sig_Coh{sub,band}(f) = 0;
            end
            %% SSC-Coh (without non-negativity constraint)
            % parameter optimization
            obj_Para = zeros(length(lambdaSet),length(lambdaSet),length(gammaSet),length(gammaSet));
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
                        for lambda1 = 1:length(lambdaSet)
                            for lambda2 = 1:length(lambdaSet)
                                para = [lambdaSet(lambda1) lambdaSet(lambda2) gammaSet(gamma1) gammaSet(gamma2)];
                                [alpha,beta,~,~,obj_SSCCMC] = SSC_Coh(trainData,para,alpha0,beta0,phi0,mode);
                                if sum(alpha) ~= 0 && sum(beta) ~= 0
                                    est_SSCCMC = abs(alpha'*Sxy*beta)^2/abs((alpha'*Sxx*alpha)*(beta'*Syy*beta));
                                else
                                    est_SSCCMC = 10;
                                end
                                obj_Para(lambda1,lambda2,gamma1,gamma2) = obj_Para(lambda1,lambda2,gamma1,gamma2)+...
                                    abs(est_SSCCMC-obj_SSCCMC(end))/obj_SSCCMC(end);
                            end
                        end
                    end
                end
            end
            obj_Para = obj_Para*1000;
            [max_val, position_max] = min(obj_Para(:));
            [x1,x2,x3,x4] = ind2sub(size(obj_Para),position_max);
            fixed_gamma1 = gammaSet(x3(1));
            fixed_gamma2 = gammaSet(x4(1));
            fixed_lambda1 = lambdaSet(x1(1));
            fixed_lambda2 = lambdaSet(x2(1));
            % real value
            para = [fixed_lambda1 fixed_lambda2 fixed_gamma1 fixed_gamma2];
            [~,~,~,~,obj_SSCCMC,SSCCMC_EEGTopo{sub,band}(:,f),SSCCMC_EMGTopo{sub,band}(:,f)] = SSC_Coh(data,para,alpha0,beta0,phi0,mode);
            obj_SSCCMC = obj_SSCCMC(end);
            SSCCMC_Coh{sub,band}(f) = obj_SSCCMC;
            % permuted value
            rand_SSCCMC = zeros(1,PermuteNum);
            for r = 1:PermuteNum
                randData.Y = data.Y(order(r,:),:);
                [~,~,~,~,SSCCMCbuffer] = SSC_Coh(randData,para,alpha0,beta0,phi0,mode);
                rand_SSCCMC(r) = SSCCMCbuffer(end);
            end

            rand_SSCCMC = sort(rand_SSCCMC,'descend');
            if obj_SSCCMC > rand_SSCCMC(floor(PermutePoint))
                SSCCMC_Sig_Coh{sub,band}(f) = obj_SSCCMC-mean(rand_SSCCMC);
            else
                SSCCMC_Sig_Coh{sub,band}(f) = 0;
            end
        end
    end
end

savename = strcat(pwd,'\Figure\Healthy_all.mat');
save(savename,'faxis',...
    'CCMC_Sig_Coh','CCMC_EEGTopo','CCMC_EMGTopo','CCMC_Coh',...
    'SCCMC_Sig_Coh','SCCMC_EEGTopo','SCCMC_EMGTopo','SCCMC_Coh',...
    'SSCCMC_Sig_Coh','SSCCMC_EEGTopo','SSCCMC_EMGTopo','SSCCMC_Coh')

