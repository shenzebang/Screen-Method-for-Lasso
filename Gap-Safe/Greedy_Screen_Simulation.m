n = 500;
p = 1000;
k = 100;
noise_scale = 1e-3;
X = randn(n, p); X_norm = sqrt(sum(X.*X)); X = X./(ones(n, 1)*X_norm);
beta = randn(p, 1)*1e3;
beta(k+1:end) = 0;
noise = randn(n, 1)*noise_scale;
y = X*beta+noise;
lambda = 5;
theta = noise/lambda;
[beta_, theta_, A_rate, residual_record] = Greedy_Screen_Single(X, y, 1e-7, 1, 1e2, lambda, zeros(p, 1), zeros(n, 1));
SE = norm(X*beta_-y, 2);