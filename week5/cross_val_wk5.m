function [cv_err, cv_std] = cross_val_wk5(x,f,k)
N = size(x,1);
CV = [];

for n=1:N
    X = x;
    t = f;
    X(n,:) = [];
    t(n) = [];
    Xt = x(n,:);
    tt = f(n);
    [e,tp] = knn_multi_class(X,t,Xt,tt,k);
    CV  = [CV; e]; 
end

cv_err = mean(CV);
cv_std = ((N-1)/N)*sum((CV-cv_err).^2);
