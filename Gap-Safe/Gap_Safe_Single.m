function [beta, theta, A_rate] = Gap_Safe_Single(X, y, eps, f, maxit, lambda, beta_0, theta_0)
% [beta, theta, A_rate] = Gap_Safe_Single(X, y, eps, f, maxit, lambda, beta_0, theta_0)
% input :
% X : dictionary matrix
% y : response vector
% eps : convergence criterion (duality gap)
% f : frequency of screening procedure
% maxit : maximum iteration
% lambda : regularization parameter
% beta_0 : initial primal variable
% theta_0 : initial dual variable
% output:
% beta : primal variable
% theta : dual variable
% A_rate : proportion of active variables at each iteration
[~, p] = size(X);
beta = beta_0;
theta = theta_0;
A_rate = ones(maxit, 1);
y_norm = norm(y, 2);
ff = @(beta) .5*norm(X*beta-y, 2)^2+lambda*norm(beta, 1); % primal objective
dff = @(theta) .5*y_norm^2 - lambda^2*norm(theta-y/lambda, 2)^2;

Active_Set = 1:p;
for it = 1:maxit
    if mod(it, 10) == 1
        disp(it);
        disp(ff(beta));
    end
    if ff(beta) - dff(theta) < eps
        break;
    end
    residual = y - X*beta; v = max(abs(X'*residual)); 
    alpha = min(max(-1/v, y'*residual/lambda/norm(residual, 2)^2) ,1/v);
    theta = alpha * residual;
    if f~=0 && mod(it, f) == 0
        R_ = (y_norm^2-2*ff(beta)); R_ = max(R_, 0); R_ = sqrt(R_)/lambda;
        Active_Set = Gap_Safe_Screen(X, Active_Set, y/lambda, theta, beta, R_);
    end
    A_rate(it) = length(Active_Set)/p;
    beta(Active_Set==0) = 0;
    for t = 1:length(Active_Set)
        idx = Active_Set(t);
        beta(idx) = Gap_Safe_Threshold(lambda/norm(X(:, idx), 2)^2, ...
            beta(idx) - X(:, idx)'*(X*beta-y)/norm(X(:, idx), 2)^2);
    end
end
A_rate(it:end) = A_rate(it-1);
end
function beta = Gap_Safe_Threshold(u, x)
    x_abs = abs(x);
    if x_abs > u
        beta = sign(x)*(x_abs-u);
    else
        beta = 0;
    end
end