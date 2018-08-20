function [ out ] = group_derivative( y, y_der, grp_labels, inst_grp, fn, r )
%GROUP_DERIVATIVE Summary of this function goes here
% out should be same dim as Fx1
% grp_labels = known group labels
% inst_grp = matrix of instance x group (which instances are in which groups)

	if ~exist('fn','var'); fn = 'max'; end
	if ~exist('r','var'); r = 10; end
	
    switch fn
        case 'avg'
            out = group_derivative_avg(y, y_der, grp_labels, inst_grp);
        case 'max'
            out = group_derivative_max(y, y_der, grp_labels, inst_grp);
        case 'prod'
            out = group_derivative_prod(y, y_der, grp_labels, inst_grp);
        case 'lse'
            out = group_derivative_lse(y, y_der, grp_labels, inst_grp, r);
    end

    %out = out./length(grp_labels); % normalize by total number of groups
end

