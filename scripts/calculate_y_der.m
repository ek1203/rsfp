function [ out ] = calculate_y_der( y, X, classifier )
%CALCULATE_Y_DER derivative of y
% out should have dim of NxF

	if ~exist('classifier', 'var')
		classifier = 'logistic';
	end
		
    switch classifier
        case 'logistic'
            out = logistic_derivative(y, X);
    end
            

end

