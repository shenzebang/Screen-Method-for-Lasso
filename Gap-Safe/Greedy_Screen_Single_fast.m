function [beta, theta, A_rate] = Greedy_Screen_Single_fast(X, y, eps, f, maxit, lambda, beta_0, theta_0)
% [beta, theta, A_rate] = Greedy_Screen_Single_fast(X, y, eps, f, maxit, lambda, beta_0, theta_0)
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
[n, p] = size(X);
beta = beta_0;
theta = theta_0;
A_rate = ones(maxit, 1);
y_norm = norm(y, 2);
y_lambda = y/lambda;
ff = @(beta) .5*norm(X*beta-y, 2)^2+lambda*norm(beta, 1); % primal objective
dff = @(theta) .5*y_norm^2 - lambda^2*norm(theta-y_lambda, 2)^2;

Active_Set = (1:p)';
Selected_Set = [];
for it = 1:maxit
    disp(it);
    % screen
    residual = y - X*beta;
    y_r_lambda = y_lambda'*residual; % y'*residual/lambda
    X_r = X(:, Active_Set)'*residual; 
    residual_norm = norm(residual, 2);
    v = max(abs(X_r)); 
    alpha = min(max(-1/v, y_r_lambda/residual_norm^2) ,1/v);
    theta = alpha * residual;
    if f~=0 && mod(it, f) == 0
        R_ = (y_norm^2-residual_norm^2-norm(beta, 1)*lambda*2); R_ = max(R_, 0); R_ = sqrt(R_)/lambda;
        Active_Idx = Greedy_Screen_fast(X_r, residual_norm, y_r_lambda, y_norm, alpha, R_);
        Active_Set = Active_Set(Active_Idx);
    end
    A_rate(it) = length(Active_Set)/p;
    beta(Active_Set==0) = 0;
    % check if selected coordinate is screened
    Set_Int = intersect(Active_Set, Selected_Set);
    if length(Set_Int) == length(Selected_Set)
        % no selected coordinate is screened, do not update beta
    else
        % at least one selected coordinate is screeded, update beta
        if size(Set_Int, 1) == 0
            disp('error');
        end
        [beta(Set_Int), theta, ~] = Gap_Safe_Single(X(:, Set_Int), y, eps, 0, maxit, lambda, beta(Set_Int), theta);
        beta(setdiff(Selected_Set, Set_Int)) = 0;
        residual = y - X(:, Set_Int)*beta(Set_Int);
        Selected_Set = Set_Int;
    end
    
    % select one coordinate from active set
    [~, idx] = max(abs(X(:, Active_Set)'*residual));
    if ismember(Active_Set(idx), Selected_Set) == 1
        break;
    else
        Selected_Set = union(Selected_Set, Active_Set(idx));
        [beta(Selected_Set), theta, ~] = ...
            Gap_Safe_Single(X(:, Selected_Set), y, eps, 0, maxit, lambda, beta(Selected_Set), zeros(n, 1));
    end
end
A_rate(it:end) = A_rate(it-1);
end