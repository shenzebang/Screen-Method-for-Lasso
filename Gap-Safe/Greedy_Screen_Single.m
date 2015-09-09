function [beta, theta, A_rate, residual_record] = Greedy_Screen_Single(X, y, eps, f, maxit, lambda, beta_0, theta_0)
% [beta, theta, A_rate] = Greedy_Screen_Single(X, y, eps, f, maxit, lambda, beta_0, theta_0)
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
ff = @(beta) .5*norm(X*beta-y, 2)^2+lambda*norm(beta, 1); % primal objective
dff = @(theta) .5*y_norm^2 - lambda^2*norm(theta-y/lambda, 2)^2;

Active_Set = (1:p)';
Selected_Set = [];
residual_record = zeros(n, maxit);
for it = 1:maxit
    if mod(it, 10) == 1
        disp(it);
        disp(ff(beta));
    end
    if ff(beta) - dff(theta) < eps
        break;
    end
    % screen
    residual = y - X*beta; v = max(abs(X'*residual)); 
    %
    residual_record(:, it) = residual;
    %
    alpha = min(max(-1/v, y'*residual/lambda/norm(residual, 2)^2) ,1/v);
    theta = alpha * residual;
    if alpha == y'*residual/lambda/norm(residual, 2)^2
        disp('bingo');
    end
    if f~=0 && mod(it, f) == 0
        R_ = (y_norm^2-2*ff(beta)); R_ = max(R_, 0); R_ = sqrt(R_)/lambda;
        Active_Set = Greedy_Screen(X, Active_Set, y/lambda, theta, residual, R_);
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