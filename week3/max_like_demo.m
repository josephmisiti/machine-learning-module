clear
Range = 10;
Max_Model_Order = 10;
noise_var = 100;

L=[];
x = [-Range/2:0.2:Range/2]';
N=size(x,1);
f = 5*x.^3  - x.^2 + x;
f_n = f + noise_var*randn(size(x));

[i,j]=sort(x);
X=x.^0;

for k=1:Max_Model_Order
    X=[X x.^k];
    w_hat = inv(X'*X)*X'*f_n;
    f_hat = X*w_hat;
    sigma_hat = mean((f_n - f_hat).^2);
    sigma = sigma_hat*diag(X*inv(X'*X)*X');
    
    L  = [L; -N*log(sqrt(sigma_hat)) - 0.5*N*(1 + log(2*pi))];
    figure
    plot(i,f(j),'b');
    hold on
    plot(i,f_n(j),'.k','MarkerSize',15)
    errorbar(i,f_hat(j),sigma(j),'-r.')
    hold off
    pause(1)
end

figure
plot(1:Max_Model_Order,L,'dr--');
title('Maximum Likelihood ');




