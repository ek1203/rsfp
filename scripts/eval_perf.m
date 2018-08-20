function [ out, ev, X, Y ] = eval_perf(labels, scores, posclass, type, beta)

% only 1 class
out = NaN; ev = struct(); X = NaN; Y = NaN;
if numel(unique(labels)) == 1    
    return
end

switch type
    case 'auroc'
        
        [X,Y,~, out] = perfcurve(labels==posclass, scores, posclass);
        ev.AUROC = out;
        ev.FPRs = X; ev.TPRs = Y;

    case 'aupr'
        
        [X,Y,~,out] = perfcurve(labels==posclass, scores, posclass, 'xCrit', 'reca', 'yCrit', 'prec','processnan','addtofalse');

        if length(Y) > 1
            if isnan(Y(1))
                Y(1) = Y(2);
                X(1) = 0;
            end
            out = trapz(X, Y);
        end
        
        ev.AUPR = out;
        ev.TPRs = X; ev.PPVs = Y;

    case 'fmax'    
        [ out, ev ] = eval_fmax(labels==posclass, scores,1);
        X = ev.rc; Y = ev.pr;
        
    case 'fbeta'    
        if ~exist('beta','var')
            beta = 2;
        end
        [ out, ev ] = eval_fmax(labels==posclass, scores,beta);
        X = ev.rc; Y = ev.pr;
        
    case 'micro_aupr'
        % treat everything as one pool
        [ out, ev, X, Y ] = eval_perf(labels(:), scores(:), posclass, 'aupr');   
        
    case 'macro_aupr'
        % calculate aupr for each column and then average
        % assuming dim of input labels is [sample x labels]
        perfs = nan(1,size(labels,2));
        wAnnot = find(sum(labels,1)>0);
        for w=1:numel(wAnnot)
            i = wAnnot(w);
            perfs(i) = eval_perf(labels(:,i), scores(:,i), posclass, 'aupr');
        end
        out = nanmean(perfs);
        ev = perfs;

end

out = full(out);
