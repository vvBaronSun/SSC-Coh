function Err = cal_error(Topo1,Topo2,n)

if exist('n') ~= 1
    n = 1:length(Topo1);
end

%% input (nchannel*1)
Topo11 = Topo1; Topo12 = -Topo1;
Topo11 = (Topo11 - min(Topo11))/(max(Topo11)-min(Topo11));
Topo12 = (Topo12 - min(Topo12))/(max(Topo12)-min(Topo12));
Topo2 = (Topo2 - min(Topo2))/(max(Topo2)-min(Topo2));
Topo11 = Topo11(n); Topo12 = Topo12(n);
Topo2 = Topo2(n);
Err1 = 1-abs(Topo11'*Topo2)/(norm(Topo11)*norm(Topo2));
Err2 = 1-abs(Topo12'*Topo2)/(norm(Topo12)*norm(Topo2));
Err = min(Err1,Err2);
if norm(Topo1) == 0 || norm(Topo2) == 0
    Err = 1;
end

end
