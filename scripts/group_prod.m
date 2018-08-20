function [ out ] = group_prod( y, inst_grp )
%GROUP_PROD Summary of this function goes here
% Calculates the value of the aggregate group predictions
% out should be same dim as grp_labels
% grp_labels = known group labels 
% inst_grp = matrix of instance x group (which instances are in which groups)
out = zeros(size(inst_grp,2),1);

for i=1:size(inst_grp,2)
    out(i) = 1-prod(1-y(inst_grp(:,i)>0));     % 1-prod(1-score) and loc of predicted y scores
end

out = sparse(out);

end
