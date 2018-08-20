function [ out ] = get_diff(y)
% pairwise difference
% out should be dim NxN
% 2 is from the (y_i-y_j)^2, since the derivative means it's 2*(y_i-y_j) but there's also
% another set of x2 in similarity_derivative_logistic.m code... double-counting? (from gicf code)

    %y = reshape(y, [], 1);
    out = 2*(bsxfun(@minus, y, transpose(y)));	 
    
