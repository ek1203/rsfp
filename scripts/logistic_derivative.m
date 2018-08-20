function [ out ] = logistic_derivative(y, X)
% returns the value derivative of the logistic function for a specific array of values
% out should have dim of X matrix (NxF)
% y has dim of Nx1

    y_der = reshape(y.*(1-y),[],1);
    out = bsxfun(@times, X, y_der); 