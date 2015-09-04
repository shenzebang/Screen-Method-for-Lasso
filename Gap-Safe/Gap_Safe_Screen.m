function Active_Set = Gap_Safe_Screen(X, Active_Set, y_lambda, theta, beta, R_)
    for t = 1:length(Active_Set)
        x = X(:, Active_Set(t));
        R = norm(theta - y_lambda, 2)^2 - R_^2; R = sqrt(R);
        if theta'*x + R < 1+e-10
            msg = 'screening ' + num2str(Active_Set(t));
            disp(msg);
            Active_Set(t) = 0;
        end
    end
    Active_Set = Active_Set(Active_Set~=0);
end