function [cv_err, cv_std] = cross_val(x,f)
N = size(x,1);
CV = [];

for n=1:N
    X = x;
    t = f;
    X(n,:) = [];
    t(n) = [];
    Xt = x(n,:);
    tt = f(n);
    w_hat = inv(X'*X)*X'*t;
    f_t = Xt*w_hat;    
    CV  = [CV; (f_t - tt).^2]; 
end

cv_err = mean(CV);
cv_std = sqrt(((N-1)/N)*sum((CV - cv_err).^2));
