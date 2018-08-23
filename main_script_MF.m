addpath('scripts')
rng(1)

% Setup directories
data_dir = 'data';
results_dir = 'results';
mkdir(results_dir)

% Load training annotations
load([data_dir filesep 'annotations_data.mat'],'trainProts_label','trainRegions_label','trainProts','trainRegions')
load([data_dir filesep 'MF_data.mat'])

trainProts_label = trainProts_label(trainProts,goTerms);
trainRegions_label = trainRegions_label(trainRegions,goTerms);

% Select weights for gridsearch
w1_range = [1e-2 1e0];
w2_range = [0 1e-2 1e0];
lambda_range = [1e-2 1e-1 1e0];

% Select features to train on
feat_types = {'key_norm','ipr_norm','sig_norm','kmer_norm'};
	
% To speed up calculations, you can modify and run multiple instances of this script
% on a high performance computing cluster
for f=1:numel(feat_types)
	feat_type = feat_types{f};
    feat_dir = strjoin({'results','MF',feat_type}, filesep);

	mkdir(feat_dir)
	
	% Load necessary data for given feature type	
	features = load([data_dir filesep 'features_data.mat'], ['*Feat_' feat_type],'regionProt');
	eval(['protFeat = double(features.protFeat_' feat_type ');'])
	eval(['regionFeat = double(features.regionFeat_' feat_type ');'])
	regionProt = features.regionProt;
	eval(['theta_baseline = theta_baseline_' feat_type ';'])
	clear('features')

	copyfile(['scripts' filesep 'gridsearch.m'],feat_dir)
	run([feat_dir filesep 'gridsearch.m'])
end

