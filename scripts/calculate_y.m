function [ out ] = calculate_y(x, theta, classifier)
% calculate predicted score from x and theta
% out should be of dim Nx1
% x has dim of NxF
% theta has dim of Fx1

	if ~exist('classifier', 'var')
		classifier = 'logistic';
	end
		
	switch classifier
		case 'logistic'
            if size(x,2) < length(theta)
                x = [ones(size(x,1),1), x];
            end
			theta = reshape(theta, [], 1);
			out = sigmoid(x*theta);
	end

    out = round(out,3);