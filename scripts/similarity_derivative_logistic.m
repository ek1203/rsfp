function [ out ] = similarity_derivative_logistic(y, y_der_matrix, W_ij)
% Calculates the value of the derivative of the first term in the cost function that has to do with similarity.    Returns an array
% out should be same dim as theta (column vector)
% y_der_matrix has dim of X matrix
% W_ij has dim length(y) x length(y)

    y_diff = get_diff(y);	% y_diff has dim length(y) x length(y)
    x = W_ij.*y_diff;
    a = (2 * transpose(x)) * y_der_matrix;
    b = (2 * x) * y_der_matrix;
    out = sum(a, 1) - sum(b, 1);
	
	% reshape to give column vector
	out = reshape(out, [], 1);
	
    