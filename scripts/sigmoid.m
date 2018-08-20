function g = sigmoid(z)
% Computes sigmoid of z
% g should be of dim length(x) x 1 (column vector)

g = 1.0 ./ (1.0 + exp(-z));
