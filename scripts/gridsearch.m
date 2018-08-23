% Get cutoff values to set similarity scores below 95 percentile to 0
%rand_pairsim = get_sim_matrix_vec(regionFeat(randsample(size(regionFeat,1), 10000),:), 'cos');
%[f,x] = ecdf(rand_pairsim);
%cutoff = x(find(f>=0.95,1));
switch feat_type
    case 'key_norm'
        cutoff = 0.0778;
    case 'kmer_norm'
        cutoff = 0.0546;
    case 'ipr_norm'
        cutoff = 0;
    case 'sig_norm'
        cutoff = 0.0301;
end

% Suppresses warnings from pdist when using cosine metric
warning('off','stats:pdist:ZeroPoints')

% Set feature columns without occurrences in regions to 0
feat_inds = find(sum(regionFeat,1)==0)+1;

for i=1:numel(goTerms)
    fprintf('GO term %u/%u\n', i, numel(goTerms))
    sgd_settings = struct();
    sgd_settings.epochs = 50;
    sgd_settings.momentum_value = 0.9;
    sgd_settings.min_var = 0.00005;
    sgd_settings.n_var = 15;
    sgd_settings.min_epoch = 3;
    
    sgd_settings.agg_fcn = 'max';
    sgd_settings.decay = 0.1;
    sgd_settings.decay_n = 1;
    sgd_settings.eval_metric = 'aupr';
    sgd_settings.feat_type = feat_type;
    
    sgd_settings.cutoff = cutoff;
    sgd_settings.similarity_fcn = 'cos';
    
    
    sgd_settings.reg_fcn = 'l2';
    sgd_settings.lr = 0.1;
    
    theta_init = theta_baseline{i};
    theta_init(feat_inds) = 0;
    
    for w1=1:numel(w1_range)
        sgd_settings.w1 = w1_range(w1);
        
        for w2=1:numel(w2_range)
            sgd_settings.w2 = w2_range(w2);
            
            for lambda=1:numel(lambda_range)
                sgd_settings.lambda = lambda_range(lambda);
                
                filename = [strjoin({ num2str(goTerms(i)), feat_type, sgd_settings.reg_fcn, num2str(w1), num2str(w2), num2str(lambda)}, '_')];
                
                if ~exist(strcat(filename, '_last.mat'),'file') && ~exist(strcat(filename, '.mat'),'file')
                    save(strcat(filename, '.mat'),'sgd_settings')
                    theta = stochastic_gradient_descent_train(regionFeat(trainRegions,:), regionFeat(trainRegions,:), trainRegions_label(:,i), ...
                        trainProts_label(:,i), regionProt(trainRegions, trainProts), filename, sgd_settings, theta_init);
                end
            end
        end
    end
end
