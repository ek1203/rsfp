function [ out ] = group_pred( y, inst_grp, fn, r )
%GROUP_PRED Summary of this function goes here
% out should be same dim as grp_labels
% grp_labels = known group labels
% inst_grp = matrix of instance x group (which instances are in which groups)

	if ~exist('fn','var'); fn = 'max'; end
	if ~exist('r','var'); r = 10; end
	
    switch fn
        case 'avg'
            out = group_avg(y, inst_grp);
        case 'max'
            out = group_max(y, inst_grp);
        case 'prod'
            out = group_prod(y, inst_grp);
    end

end

