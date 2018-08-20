function [ out ] = group_derivative_max( y, y_der, grp_labels, inst_grp )
%GROUP_DERIVATIVE_MAX Summary of this function goes here
% Calculates the value of the derivative of the second term in the cost function, which is based on the group cost. Again returns array
% out = dim of group x features
% y = instance scores
% y_der should have dim of instance x features
% grp_labels = known group labels 
% inst_grp = matrix of instance x group (which instances are in which groups)

n_grp = length(grp_labels);
out = zeros(n_grp, size(y_der,2));

for i=1:n_grp
    instance_inds = find(inst_grp(:,i)>0);    % index of instances in group
    [max_val, max_ind] = max(y(instance_inds));     % max value and loc of predicted y scores
    out(i,:) = 2 * (max_val - grp_labels(i)) * y_der(instance_inds(max_ind),:); % multiply by the derivative of the max instance?
end

end

