function x = multi_var_gauss_sampler(m,C)
D = size(C,1);
x = m + chol(C)'*randn(D,1);