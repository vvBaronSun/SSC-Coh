function [newX,U,dim]=Dimreduction(X,channel,info)
CMatrix = real(X'*X);
[U,S,~] = svd(CMatrix);
eigenval = diag(S);
infoFuc = 0;
for i = 1:channel %确定EEG信息量
    infoFuc = infoFuc + eigenval(i);
    if(infoFuc > info * sum(eigenval))
        dim = i;
        break
    end
end
U = U(:,1:dim);
newX = X*U;

end