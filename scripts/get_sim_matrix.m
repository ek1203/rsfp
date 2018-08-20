function [ out_mat, out_vec ] = get_sim_matrix(x, fcn, cutoff)
% generate similarity matrix and set values below cutoff to 0
if strcmp(fcn, 'cos')
    x = x(:,sum(x,1)>0);    % remove all-zero columns for faster calculations
    out_vec = pdist(x, fcn);
else
    out_vec = pdist(x, fcn);
end 

out_vec(isnan(out_vec)) = 0;
out_mat = 1 - squareform(out_vec);
out_mat(out_mat <= cutoff) = 0; 
out_vec = 1 - out_vec;
out_vec(out_vec <= cutoff) = 0; 
