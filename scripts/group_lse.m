function [ out ] = group_lse( y, inst_grp, r )
%GROUP_LSE Summary of this function goes here
% Calculates the value of the aggregate group predictions
% out should be same dim as grp_labels
% grp_labels = known group labels 
% inst_grp = matrix of instance x group (which instances are in which groups)
% as formulated in Revisiting Multiple Instance Neural Networks (Wang 2016)
% more like LogMeanExp
out = zeros(size(inst_grp,2),1);

for i=1:size(inst_grp,2)
    %out(i) = log(sum(exp(y(inst_grp(:,i)>0))));     % max value and loc of predicted y scores
    out(i) = r^-1*log(mean(exp(r*y(inst_grp(:,i)>0))));     % max value and loc of predicted y scores

end

%out(out>1) = 1;	% values go beyond 1
out = sparse(out);

end
