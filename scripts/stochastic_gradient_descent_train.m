function [ theta, perf_train ] = stochastic_gradient_descent_train(inst_feat_train, inst_feat_train_sim, inst_labels_train, grp_labels_train, ...
    inst_grp_train, output_name, sgd_settings, theta_init)
fprintf('%s\n',mfilename('fullpath'))
fprintf('Output filename: %s\n',output_name)

outFile = strcat(output_name,'.mat');
outFile_last =strcat(output_name, '_last.mat');

% add columns of ones for each feature matrix for first term of logistic regression coefficient
inst_feat_train = [ones(size(inst_feat_train,1),1), inst_feat_train];
F = size(inst_feat_train,2);

t = cputime;

% default settings
if isempty(sgd_settings)
    sgd_settings.epochs = 50;
    sgd_settings.momentum_value = 0.9;
    sgd_settings.min_var = 0.00005;
    sgd_settings.n_var = 15;
    sgd_settings.min_epoch = 3;
    
    sgd_settings.agg_fcn = 'max';
    sgd_settings.decay = 0.1;
    sgd_settings.decay_n = 1;
    sgd_settings.eval_metric = 'aupr';
    sgd_settings.reg_fcn = 'l2';        % options: {'l1','l2'}
    sgd_settings.cutoff = 0;
    sgd_settings.similarity_fcn = 'cos';
    sgd_settings.gamma = 0.1;
    sgd_settings.lr = 0.1;
    
    sgd_settings.w1 = 1e-1;        % weight of similarity_cost
    sgd_settings.w2 = 1e-1;        % weight of instance_cost
    sgd_settings.lambda = 1e-3;        % weight of regularization_cost
end
disp(sgd_settings)

subset_grp_inds = [];
subset_inst_inds = [];
inst_feat_s = [];
inst_feat_sim_s = [];
inst_labels_s = [];
grp_labels_s = [];
inst_grp_s = [];

velocity = zeros(F,1);
terminate = false;
total_iterations = 0;

theta = theta_init;
perfs_train = eval_groups();
best_perf = perfs_train;
best_theta = theta;

fprintf('Initial training set perf = %f\n',full(best_perf))

for epoch=1:sgd_settings.epochs
    % split data into batches
    [rand_inds, batch_size] = split_data(grp_labels_train);
    n_batches = numel(rand_inds);
    if terminate; break; end
    
    fprintf('Epoch = %u/%u, Number of subsets: %u, batch size = %u\n', epoch, sgd_settings.epochs, n_batches, batch_size)
    
    % go through entire training set
    for iter=1:n_batches
        total_iterations = total_iterations + 1;
        get_new_subset(1:numel(grp_labels_train));
        
        % N x F = number of samples x number of features
        N = size(inst_feat_s, 1);
        
        % gradient descent
        % similarity matrix does not have to be from inst_feat_train
        W_ij = get_sim_matrix(inst_feat_sim_s, sgd_settings.similarity_fcn, sgd_settings.cutoff);
        
        % calculate y_hat and derivative
        % y = predicted instance scores
        inst_scores_s = calculate_y(inst_feat_s, theta);
        inst_scores_der_s = calculate_y_der(inst_scores_s, inst_feat_s);    % dim = inst x features
        group_scores_der_s = group_derivative(inst_scores_s, inst_scores_der_s, grp_labels_s==1, inst_grp_s, sgd_settings.agg_fcn); % dim = groups x features
        
        % calculate cost
        similarity_cost = similarity_derivative(inst_scores_s, inst_scores_der_s, W_ij);
        group_cost = reshape(sum(group_scores_der_s,1),[],1);
        
        % only for positive labels (since negative labels can be positive unlabeled)
        instance_cost = zeros(size(theta));
        inst_labels_s_pos = inst_labels_s == 1;
        if any(inst_labels_s_pos)
            instance_cost = instance_derivative(inst_scores_s(inst_labels_s_pos), inst_scores_der_s(inst_labels_s_pos,:), inst_labels_s(inst_labels_s_pos))./sum(inst_labels_s_pos);
        end
        
        % if L2-regularization term specified
        regularization_cost = zeros(size(theta));
        if strcmp(sgd_settings.reg_fcn, 'l2')
            regularization_cost = [0; regularization_derivative(theta(2:end))];  % theta0 not calculated with regularization
        end
        
        theta_der = (1/length(grp_labels_s)) * group_cost ...
            + (sgd_settings.w1/N^2) * similarity_cost ...
            + sgd_settings.w2 * instance_cost ...
            + (sgd_settings.lambda/length(grp_labels_s)) * regularization_cost;
        
        % update velocity and theta
        learning_rate_decay = ((1-sgd_settings.decay)^(floor(epoch/sgd_settings.decay_n)));     % decay after decay_n epochs
        velocity = sgd_settings.momentum_value * velocity - sgd_settings.lr * learning_rate_decay * theta_der;
        theta = theta + velocity;
        
        % if L1-regularization term specified
        if strcmp(sgd_settings.reg_fcn, 'l1')
            theta_tmp = theta(2:end);
            theta_tmp(theta_tmp > 0) = max(0, theta_tmp(theta_tmp>0) - (sgd_settings.lambda/N)*sgd_settings.lr * learning_rate_decay);
            theta_tmp(theta_tmp < 0) = min(0, theta_tmp(theta_tmp<0) + (sgd_settings.lambda/N)*sgd_settings.lr * learning_rate_decay);
            theta(2:end) = theta_tmp;
        end
        
        % print progress
        if mod(total_iterations, 50) == 0
            perf_train = eval_groups();
            perfs_train = [perfs_train; perf_train];
            if perf_train > best_perf  % save best theta, based on training set
                best_perf = perf_train;
                best_theta = theta;
            end
            
            [variance,trend] = terminate_conditions(perfs_train);
            fprintf('--- Iter = %u, curr_train_perf = %.2f, norm = %.3f, var = %.5f, trend = %.5f, lr_decay = %.3f\n', total_iterations, full(perf_train), norm(theta), full(variance), full(trend), learning_rate_decay)
        end
    end
end

e = cputime-t;

perf_train = eval_groups();
perfs_train = [perfs_train; perf_train];

save(outFile_last, 'e', 'theta','theta_init','sgd_settings','perfs_train','best_theta','best_perf','total_iterations','epoch','n_batches')
delete(outFile)

fprintf('Elapsed time: %f, output_file = %s, Total iterations = %u, perf_init = %.2f, perf_final = %.2f, norm_init = %.3f, norm = %.3f\n',e, outFile_last, total_iterations, full(perfs_train(1)), full(perfs_train(end)), norm(theta_init), norm(theta))

    function [ variance, trend ] = terminate_conditions(perfs)
        % conditions to terminate
        % variance of last n_var perfs (if more than n_var in length)
        % i.e. the perfs are consistent with the last n_var random subsets of
        % data in training
        % also continue if performance is increasing
        perfs(isnan(perfs)) = [];
        
        if length(perfs) > sgd_settings.n_var
            perfs = perfs(end-sgd_settings.n_var+1:end);
        end
       
        variance = var(perfs, 0, 1);
        trend = perfs(end)-perfs(1);
        terminate = (epoch > sgd_settings.min_epoch && variance < sgd_settings.min_var && trend <= 0);
    end

    function [ ] = get_new_subset(subset_inds)
        % get new batch
        subset_grp_inds = subset_inds(rand_inds{iter});
        subset_inst_inds = sum(inst_grp_train(:,subset_grp_inds),2)>0;
        inst_feat_s = inst_feat_train(subset_inst_inds,:);
        inst_feat_sim_s = inst_feat_train_sim(subset_inst_inds,:);
        inst_labels_s = inst_labels_train(subset_inst_inds);
        grp_labels_s = grp_labels_train(subset_grp_inds);
        inst_grp_s = inst_grp_train(subset_inst_inds,subset_grp_inds);
    end

    function [ perf_train ] = eval_groups()
        inst_scores_train = calculate_y(inst_feat_train, theta);
        grp_scores_train = group_pred(inst_scores_train, inst_grp_train, sgd_settings.agg_fcn);
        perf_train = eval_perf(grp_labels_train, grp_scores_train, 1, sgd_settings.eval_metric);
    end

end
