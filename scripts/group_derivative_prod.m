function [ out ] = group_derivative_prod( y, y_der, grp_labels, inst_grp )
%GROUP_DERIVATIVE_PROD Summary of this function goes here
% Calculates the value of the derivative of the second term in the cost function, which is based on the group cost. Again returns array
% out = dim of group x features
% y = instance scores
% y_der should have dim of instance x features
% grp_labels = known group labels 
% inst_grp = matrix of instance x group (which instances are in which groups)
% based on: https://en.wikipedia.org/wiki/Product_rule#A_product_of_more_than_two_factors

grp_score = group_prod( y, inst_grp );
n_grp = length(grp_labels);
out = zeros(n_grp, size(y_der,2));

for i=1:n_grp
    instance_inds = inst_grp(:,i)>0;    % index of instances in group
    term1 = prod(1-y(instance_inds));
    term2 = sum(bsxfun(@rdivide, y_der(instance_inds,:), 1-y(instance_inds)),1);
    term2(isnan(term2)) = 0;
    grp_i_der = term1*term2;
    out(i,:) = 2 * (grp_score(i) - grp_labels(i)) * grp_i_der;
end

end

