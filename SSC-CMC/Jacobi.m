function y_diff1 = Jacobi(alpha,beta,phi,opts) %LPhi微分函数
% --------------------------------------------------------------------
% Input:
%       - alpha, fixed variable
%       - beta, fixed variable
%       - phi: independent variable
%       - opts: given parameter
%
% Output:
%       - y_diff1, first order differential of optFunc
%
%---------------------------------------------------------------------
% Author: Jingyao Sun, sunjy22@mails.tsinghua.edu.cn
% Date created: July-30-2024
% @Tsinghua Univertity.
% --------------------------------------------------------------------

Dt = 0.001; % step size 0.001
phi_left = (phi-0.005):0.001:phi; % 向左取5个
phi_right = (phi+0.001):0.001:(phi+0.005); % 向右取5个
phiAll = [phi_left phi_right];
OptFuncValue = zeros(1,size(phiAll,2));
for i = 1:size(phiAll,2)
    OptFuncValue(i) = optFunc(alpha,beta,phiAll(i),opts); %得到OptFunc函数值
end
[dy1,~] = diff_ctr(OptFuncValue,Dt,1);
y_diff1 = dy1(ceil(size(dy1,2)/2)); %求在phi点的一阶微分
end