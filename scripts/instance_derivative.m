function [ out ] = instance_derivative(y, y_der, y_labels)
% out should have dim of Fx1
% y, y_labels has dim of Nx1
% y_der has dim of NxF

	out = 2 * transpose(y_der)*(y-y_labels);
	%out = reshape(out, [], 1);