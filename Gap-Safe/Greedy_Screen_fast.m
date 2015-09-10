function Active_Set = Greedy_Screen_fast(X_r, residual_norm, r_y_lambda, y_norm, alpha, R_)
% function Active_Set = Greedy_Screen_fast(X_r, residual_norm, r_y_lambda, y_norm, alpha, R_)
    a = abs(X_r); b = residual_norm^2; c = r_y_lambda; d = y_norm^2 - R_^2;
    Active_Set = 1:length(X_r);
    for t = 1:length(X_r)
        fd = @(x) abs(x)*a(t) + sqrt(b*x^2-2*c*x+d);
        if fd(alpha) < 1-1e-10
            Active_Set(t) = 0;
            disp('screen');
        end
    end
    Active_Set = Active_Set(Active_Set~=0);
end