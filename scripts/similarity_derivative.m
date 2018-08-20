function [ out ] = similarity_derivative( y, y_der_matrix, W_ij, classifier )
%SIMILARITY_DERIVATIVE Summary of this function goes here
% out should be same dim as Fx1
% y_der_matrix has dim of NxF
% W_ij has dim NxN

	if ~exist('classifier', 'var')
		classifier = 'logistic';
	end
		
    switch classifier
        case 'logistic'
            out = similarity_derivative_logistic(y, y_der_matrix, W_ij);
    end
            
            

end

