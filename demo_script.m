addpath('scripts')
rng(1)

data_dir = 'data';
results_dir = 'results';
mkdir(results_dir)

load([data_dir filesep 'demo_data.mat'])

% Suppresses warnings from pdist when using cosine metric
warning('off','stats:pdist:ZeroPoints')

fprintf('Running optimization for demo dataset\n')
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
sgd_settings.feat_type = 'key_norm';

sgd_settings.cutoff = 0.0778;
sgd_settings.similarity_fcn = 'cos';
sgd_settings.reg_fcn = 'l2';
sgd_settings.lr = 0.1;

sgd_settings.w1 = 1e-2;   
sgd_settings.w2 = 1e-2;   
sgd_settings.lambda = 1e-1;

filename = [ results_dir filesep 'demo_output' ];

theta_init = zeros(size(regionFeat,2)+1,1);
theta = stochastic_gradient_descent_train(regionFeat, regionFeat, trainRegions_label, trainProts_label, ...
    regionProt, filename, sgd_settings, theta_init);

