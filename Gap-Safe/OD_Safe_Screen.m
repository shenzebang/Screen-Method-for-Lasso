function Active_Set = OD_Safe_Screen(X, Active_Set, y_lambda, residual, beta, R_)
% function Active_Set = OD_Safe_Screen(X, Active_Set, y_lambda, residual, beta, R_)
    a = abs(X'*residual); b = norm(residual, 2)^2; c = residual'*y_lambda; d = norm(y_lambda, 2)^2 - R_^2;
    v = 1/max(a);
    alpha = min(max(-v, y_lambda'*residual/norm(residual, 2)^2) ,v);
    for t = 1:length(Active_Set)
        idx = Active_Set(t);
        fd = @(x) abs(x)*a(idx) + sqrt(x^2*b-2*x*c+d);
        [~, fval] = fminbnd(fd, -v, v);
%        disp(fd(alpha)-fval);
        if fd(alpha) < 1+1e-10
            msg = strcat('screening ', num2str(Active_Set(t)));
            disp(msg); % disp(abs(theta'*x) + R);
            Active_Set(t) = 0;
        end
    end
    Active_Set = Active_Set(Active_Set~=0);
end