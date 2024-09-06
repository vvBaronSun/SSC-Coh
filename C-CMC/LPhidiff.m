function y_diff1 = LPhidiff(CMatrix,EEGchannel,EMGchannel,Phi) %LPhi微分函数
%CMatrix     totalchannels*totalchannels*frequency      the cross-spectrum block matrix between sets(EEG and EMG)
%EEGchannel     the number of EEG channels
%EMGchannel     the number of EMG channels
%feature     1*frequency   the maximum of Coherence
%EEGweight   EEGchannels*frequency  the weight of EEG channels
%EMGweight   EMGchannels*frequency  the weight of EMG channels 
%Phi        自变量
Dt = 0.001; %step size 0.001
y_left = (Phi-0.005):0.001:Phi; %向左取5个
y_right = (Phi+0.001):0.001:(Phi+0.005); %向右取5个
y = [y_left y_right];
LPhivalue = zeros(1,size(y,2));
for i = 1:size(y,2)
    [LPhivalue(i),~,~,~]=LPhi(CMatrix,EEGchannel,EMGchannel,y(i)); %得到LPhi函数值
end
[dy1,~] = diff_ctr(LPhivalue,Dt,1);
y_diff1 = dy1(ceil(size(dy1,2)/2)); %求在maxPhi点的一阶微分
% [dy2,dx2]=diff_ctr(LPhivalue,Dt,2);
% flag2 = dx2 == Phi;
% y_diff2=dy2(flag2); %求在maxPhi点的二阶微分