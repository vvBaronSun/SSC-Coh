function [fdata,faxis] = FourierTransform(data,samplerate)
% Calculate the Fourier Transform for the data after pre-processing
% Input:
% ---------------------
% data   - pre-processed signal (time*channel*trial)
% samplerate - sample rate/Hz
%
% Output:
% -------------------
% fdata   - signal after Fourier Transform
%
%
%% initialize
[timenum,channel,trialnum] = size(data);
faxis = samplerate * (0:(timenum/2)) / timenum;
fdata = zeros(length(faxis),channel,trialnum);
%% calculate Fourier Transform
for trial = 1:trialnum
    Xdata = fft(data(:,:,trial))/timenum;
    fdata(:,:,trial) = Xdata(1:timenum/2+1,:);
    fdata(2:end-1,:,trial) = 2*fdata(2:end-1,:,trial);
end

end