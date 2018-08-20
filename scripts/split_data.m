function [ out, batch_size ] = split_data(grp_labels)
% Stratified sampling of data to keep relative proportions of positive and negative labels.
% Replace positive examples in case there are too few.
% Automatically determine batch size based on number of positive examples.

pos_inds = find(grp_labels == 1);
neg_inds = find(grp_labels ~= 1);  
n_groups = length(grp_labels);
batch_size = max(128, ceil(n_groups/numel(pos_inds)));

n_sets = floor(n_groups/batch_size);
out = cell(n_sets, 1);

p = ceil(numel(pos_inds)/n_sets);
n = floor(numel(neg_inds)/n_sets);

for i=1:n_sets
    pos = datasample(pos_inds, p, 'Replace', true);    
    neg = datasample(neg_inds, n, 'Replace', false); 
    neg_inds = setdiff(neg_inds, neg);
    out{i} = [pos; neg];
end
