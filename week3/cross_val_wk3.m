function [cv_err, cv_std] = cross_val_wk3(x,f,a,b,c)
N = size(x,1);
CV = [];

for n=1:N
    X = x;
    t = f;
    X(n,:) = [];
    t(n) = [];
    Xt = x(n,:);
    tt = f(n);
    K=kernel_func(X,X,'gauss',c,c)';
    Kt=kernel_func(X,Xt,'gauss',c,c)';
    f_t = Kt*inv(K'*K + (a/b)*eye(N-1))*K'*t;
    CV  = [CV; (f_t - tt).^2]; 
end

cv_err = mean(CV);
cv_std = ((N-1)/N)*sum((CV-cv_err).^2);
