function [LPhi,DPhi,CAA_inv,CBB_inv,CAA,CBB]=LPhi(newCMatrix,EEGdim,EMGdim,Phi)
% CMatrix     totalchannels*totalchannels      the cross-spectrum block matrix between sets(EEG and EMG)
% EEGchannel     the number of EEG channels
% EMGchannel     the number of EMG channels
% feature     1*frequency   the maximum of Coherence
% EEGweight   EEGchannels*frequency  the weight of EEG channels
% EMGweight   EMGchannels*frequency  the weight of EMG channels 
% Phi        自变量
%% calculate DPhi
if (EEGdim+EMGdim) ~= length(newCMatrix)
    error('The dimension is incorrect!')
end
CAA = real(newCMatrix(1:EEGdim,1:EEGdim));
CAA_inv = CAA^(-1/2);
CBB = real(newCMatrix((EEGdim+1):end,(EEGdim+1):end));
CBB_inv = CBB^(-1/2);
CAB = newCMatrix(1:EEGdim,(EEGdim+1):end);
PhiCAB = real(exp(-Phi*1i)*CAB);
DPhi = CAA_inv*PhiCAB*CBB_inv;
LPhi = max(eig(DPhi*DPhi'));

end