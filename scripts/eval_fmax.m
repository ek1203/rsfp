function [fmax, ev] = eval_fmax(actual, predicted, beta, tau)
%% 
% ev.pr = precision 
% ev.rc = recall
% ev.cov = coverage (fraction of benchmark proteins on which the method made any predictions)
% ev.fmax = max fmeasure

if ~exist('tau','var')
    tau = 0.00:0.01:1.00; 
end

if isempty(beta)
    beta = 1;
end

k = numel(tau);

% make sure it is a row if it is a single column
if size(actual,2)==1; actual = transpose(actual); end
if size(predicted,2)==1; predicted = transpose(predicted); end

% evaluate only on samples with at least 1 positive annotation
withAnnots = sum(actual,2)>0;
actual = actual(withAnnots,:);
predicted = predicted(withAnnots,:);
ev.cov = nnz(sum(predicted,2))/size(actual,1);

for i = 1 : k
    [ev.pr(i), ev.rc(i)] = precision_recall(actual, predicted, tau(i));
end

ev.fbeta = ((1+beta^2)*ev.pr.*ev.rc)./((beta^2.*ev.pr)+ev.rc);
fmax = max(ev.fbeta);
threshold = find(ev.fbeta==fmax,1,'last');

ev.fmax = fmax;
ev.tmax = tau(threshold);
ev.tau = tau;
ev.neval = size(actual,1);

    function [pr_val, rc_val] = precision_recall(actual, predicted, tau)
        % precision and recall for each row
        predicted = logical(predicted>=tau);
        TP = sum(actual.*predicted,2);
        pr_val = nanmean(TP./sum(predicted,2));
        rc_val = mean(TP./sum(actual,2));
    end


end
