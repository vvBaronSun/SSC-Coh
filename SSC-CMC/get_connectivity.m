function L = get_connectivity(data)
% --------------------------------------------------------------------
% Calculate structured penalty matrix
% --------------------------------------------------------------------
% Input:
%       - data, EEG/EMG complex data matrix, n (sample) * sensors
% Output:
%       - Penalty, Laplacian matrix (diagnonal matrix - adjacency matrix),
%       sensors * sensors


n_sample = size(data,1);
sensors = size(data,2);
CMatrix = zeros(sensors,sensors);


for i = 1:sensors
    for j = 1:sensors
        X = data(:,i); Y = data(:,j);
        CMatrix(i,j) = abs(X'*Y).^2/(abs(X'*X)*abs(Y'*Y));
    end
end

diag_CMatrix = diag(diag(CMatrix));
A = CMatrix - diag_CMatrix;
D = diag(sum(A,2));
L = D - A;
end
