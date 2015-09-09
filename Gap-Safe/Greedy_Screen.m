function Active_Set = Greedy_Screen(X, Active_Set, y_lambda, theta, residual, R_)
%
%   sqrt(alpha^2*norm(residual)^2-2*alpha*residual'*y_lambda+norm(y_lambda)^2-R_^2)
    a = abs(X'*residual); b = norm(residual, 2)^2; c = residual'*y_lambda; d = norm(y_lambda, 2)^2 - R_^2;
    v = 1/max(a);
    alpha = min(max(-v, y_lambda'*residual/norm(residual, 2)^2) ,v);
%
    for t = 1:length(Active_Set)
        x = X(:, Active_Set(t));
        R = norm(theta - y_lambda, 2)^2 - R_^2; R = sqrt(R);
        %
        idx = Active_Set(t);
        fd = @(x) abs(x)*a(idx) + sqrt(b*x^2-2*c*x+d);
%        [x_opt, fval] = fminbnd(fd, -v, v);
%        if fval - (abs(theta'*x) + R) < 0
%            disp(fval - (abs(theta'*x) + R));
%        end
        %
        
        if abs(theta'*x) + R < 1-1e-10
%            msg = strcat('screening ', num2str(Active_Set(t)));
%            disp(msg); % disp(abs(theta'*x) + R);
            Active_Set(t) = 0;
        end
    end
    Active_Set = Active_Set(Active_Set~=0);
end