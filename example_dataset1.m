clear,clc
close all

addpath("SSC-CMC\")
addpath("C-CMC\")
addpath("FISTA-master\")
addpath("Dataset1\")

% load data
load('Simdata.mat') % 1 0.5 0.1 0.05

% Initialize
SNRNum = size(EEGData,1);
TrialNum = [25 50 75 100];
PermuteNum = 500; PermutePoint = PermuteNum*0.05;

% Parameter set (first optimize gamma, then lambda)
lambdaSet = [0.01 0.1 1 10];
gammaSet = [0.01 0.1 1 10];
mode = false;

% result matrix
subNum = 1;
%
CCMC_Sig_Coh = zeros(length(TrialNum),subNum); CCMC_Error = zeros(length(TrialNum),subNum,2);
CCMC_EEGTopo = cell(length(TrialNum),subNum); CCMC_EMGTopo = cell(length(TrialNum),subNum);
CCMC_Coh = zeros(length(TrialNum),subNum);
%
SCCMC_Sig_Coh = zeros(length(TrialNum),subNum); SCCMC_Error = zeros(length(TrialNum),subNum,2);
SCCMC_EEGTopo = cell(length(TrialNum),subNum); SCCMC_EMGTopo = cell(length(TrialNum),subNum);
SCCMC_Coh = zeros(length(TrialNum),subNum);
%
SSCCMC_Sig_Coh = zeros(length(TrialNum),subNum); SSCCMC_Error = zeros(length(TrialNum),subNum,2);
SSCCMC_EEGTopo = cell(length(TrialNum),subNum); SSCCMC_EMGTopo = cell(length(TrialNum),subNum);
SSCCMC_Coh = zeros(length(TrialNum),subNum);

for SNR = 1:4
    for sub = 1:subNum
        disp("Subject"+num2str(sub))
        for sample = 1:length(TrialNum)
            disp("Sample"+num2str(sample))
            % Fourier Transform
            fEEGdata = FourierTransform(EEGData{SNR,sub}(:,:,1:TrialNum(sample)),SF);
            [fEMGdata,faxis] = FourierTransform(EMGData{1,sub}(:,:,1:TrialNum(sample)),SF);
            f_tar = find(faxis == 25);

            % generate data
            data.X = squeeze(fEEGdata(f_tar,:,:))';
            data.Y = squeeze(fEMGdata(f_tar,:,:))';
            data.PX = get_connectivity(data.X);
            data.PY = get_connectivity(data.Y);
            data.Sxx = zscore(data.X)'*zscore(data.X);
            data.Sxy = zscore(data.X)'*zscore(data.Y);
            data.Syy = zscore(data.Y)'*zscore(data.Y);
            randData.X = data.X;
            randData.PX = data.PX;
            randData.PY = data.PY;
            randData.Sxx = data.Sxx;

            p = length(EEGSource); q = length(EMGSource);

            % rand order
            order = zeros(PermuteNum,TrialNum(sample));
            for i = 1:PermuteNum
                order(i,:) = randperm(TrialNum(sample));
            end

            %% C-Coh algorithm
            disp("C-Coh algorithm")
            phi0 = pi/4; info = 0.9;
            % real value
            [~,~,~,obj_CCMC,EEGtopo,EMGtopo] = C_Coh(data,phi0,info);
            CCMC_Coh(sample,sub) = obj_CCMC;

            % permuted value
            rand_CCMC = zeros(1,PermuteNum);
            for r = 1:PermuteNum
                randData.Y = data.Y(order(r,:),:);
                [~,~,~,rand_CCMC(r)] = C_Coh(randData,phi0,info);
            end

            rand_CCMC = sort(rand_CCMC,'descend');
            if obj_CCMC > rand_CCMC(floor(PermutePoint))
                CCMC_Sig_Coh(sample,sub) = obj_CCMC-mean(rand_CCMC);
            else
                CCMC_Sig_Coh(sample,sub) = 0;
            end

            % Normalize
            CCMC_EEGTopo{sample,sub} = EEGtopo;
            CCMC_EMGTopo{sample,sub} = EMGtopo;

            % Calculate error
            CCMC_Error(sample,sub,1) = cal_error(CCMC_EEGTopo{sample,sub},EEGSource);
            CCMC_Error(sample,sub,2) = cal_error(CCMC_EMGTopo{sample,sub},EMGSource);

            %% SC-Coh algorithm
            disp("SC-Coh algorithm")
            alpha0 = ones(p,1)/p;
            beta0 = ones(q,1)/q;
            phi0 = pi/4;
            % Parameter optimization
            obj_Para = zeros(length(lambdaSet),length(lambdaSet));
            indices = crossvalind('KFold',TrialNum(sample),5);
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
            obj_Para = obj_Para*1000;
            [row,col] = find(obj_Para == min(obj_Para,[],'all'));
            Fixed_lambda1 = lambdaSet(row(1));
            Fixed_lambda2 = lambdaSet(col(1));
            % real value
            para = [Fixed_lambda1 Fixed_lambda2 0 0];
            [~,~,~,~,obj_SCCMC,EEGtopo,EMGtopo] = SSC_Coh(data,para,alpha0,beta0,phi0,mode);
            obj_SCCMC = obj_SCCMC(end);
            SCCMC_Coh(sample,sub) = obj_SCCMC;
            % permuted value
            rand_SCCMC = zeros(1,PermuteNum);
            for r = 1:PermuteNum
                randData.Y = data.Y(order(r,:),:);
                [~,~,~,~,SCCMCbuffer] = SSC_Coh(randData,para,alpha0,beta0,phi0,mode);
                rand_SCCMC(r) = SCCMCbuffer(end);
            end

            rand_SCCMC = sort(rand_SCCMC,'descend');
            if obj_SCCMC > rand_SCCMC(floor(PermutePoint))
                SCCMC_Sig_Coh(sample,sub) = obj_SCCMC-mean(rand_SCCMC);
            else
                SCCMC_Sig_Coh(sample,sub) = 0;
            end

            % Normalize
            SCCMC_EEGTopo{sample,sub} = EEGtopo;
            SCCMC_EMGTopo{sample,sub} = EMGtopo;

            % Calculate error
            SCCMC_Error(sample,sub,1) = cal_error(SCCMC_EEGTopo{sample,sub},EEGSource);
            SCCMC_Error(sample,sub,2) = cal_error(SCCMC_EMGTopo{sample,sub},EMGSource);

            %% SSC-Coh
            disp("SSC-Coh algorithm")
            alpha0 = ones(p,1)/p;
            beta0 = ones(q,1)/q;
            phi0 = pi/4;
            % parameter optimization
            obj_Para = zeros(length(lambdaSet),length(lambdaSet),length(gammaSet),length(gammaSet));
            indices = crossvalind('KFold',TrialNum(sample),5);
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
            [~,~,~,~,obj_SSCCMC,EEGtopo,EMGtopo] = SSC_Coh(data,para,alpha0,beta0,phi0,mode);
            obj_SSCCMC = obj_SSCCMC(end);
            SSCCMC_Coh(sample,sub) = obj_SSCCMC;
            % permuted value
            rand_SSCCMC = zeros(1,PermuteNum);
            for r = 1:PermuteNum
                randData.Y = data.Y(order(r,:),:);
                [~,~,~,~,SSCCMCbuffer] = SSC_Coh(randData,para,alpha0,beta0,phi0,mode);
                rand_SSCCMC(r) = SSCCMCbuffer(end);
            end

            rand_SSCCMC = sort(rand_SSCCMC,'descend');
            if obj_SSCCMC > rand_SSCCMC(floor(PermutePoint))
                SSCCMC_Sig_Coh(sample,sub) = obj_SSCCMC-mean(rand_SSCCMC);
            else
                SSCCMC_Sig_Coh(sample,sub) = 0;
            end

            % Normalize
            SSCCMC_EEGTopo{sample,sub} = EEGtopo;
            SSCCMC_EMGTopo{sample,sub} = EMGtopo;

            % Calculate error
            SSCCMC_Error(sample,sub,1) = cal_error(SSCCMC_EEGTopo{sample,sub},EEGSource);
            SSCCMC_Error(sample,sub,2) = cal_error(SSCCMC_EMGTopo{sample,sub},EMGSource);

             % plot
             Ground = abs(EEGSource);
             Ground = (Ground - min(Ground(:)))/(max(Ground)-min(Ground));
             Topo1 = abs(CCMC_EEGTopo{sample,1});
             Topo1 = (Topo1 - min(Topo1(:)))/(max(Topo1)-min(Topo1));
             Topo2 = abs(SCCMC_EEGTopo{sample,1});
             Topo2 = (Topo2 - min(Topo2(:)))/(max(Topo2)-min(Topo2));
             Topo3 = abs(SSCCMC_EEGTopo{sample,1});
             Topo3 = (Topo3 - min(Topo3(:)))/(max(Topo3)-min(Topo3));
             figure,ESSim.Simulate.PlotScalp(Ground);
             figure,ESSim.Simulate.PlotScalp(Topo1);
             figure,ESSim.Simulate.PlotScalp(Topo2);
             figure,ESSim.Simulate.PlotScalp(Topo3);

        end
    end

    savename = strcat(pwd,'\Figure\Sample_',num2str(SNR),'.mat');
    save(savename,'CCMC_Sig_Coh','CCMC_Error','CCMC_EEGTopo','CCMC_EMGTopo',...
        'SCCMC_Sig_Coh','SCCMC_Error','SCCMC_EEGTopo','SCCMC_EMGTopo',...
        'SSCCMC_Sig_Coh','SSCCMC_Error','SSCCMC_EEGTopo','SSCCMC_EMGTopo',...
        'CCMC_Coh','SCCMC_Coh','SSCCMC_Coh',...
        'EEGSource','EMGSource')
end



