function [ out ] = group_derivative_avg( y, y_der, grp_labels, inst_grp )
%GROUP_DERIVATIVE_AVG Summary of this function goes here
% Calculates the value of the derivative of the second term in the cost function, which is based on the group cost. Again returns array
% out = dim of group x features
% grp_labels = known group labels 
% inst_grp = matrix of instance x group (which instances are in which groups)
% y_der should have dim of X matrix (NxF)

grp_score = group_avg( y, inst_grp );
n_grp = length(grp_labels);
out = zeros(n_grp, size(y_der,2));

for i=1:n_grp
    instance_inds = find(inst_grp(:,i)>0);    % index of instances in group
	a = grp_score(i) - grp_labels(i);
	b = sum(y_der(instance_inds,:),1) / numel(instance_inds);
	out(i,:) = 2 * a*b;
end

end

