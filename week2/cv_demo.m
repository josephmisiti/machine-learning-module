clear
Range = 10;
Nos_Samps = 50;
Nd = 100;
Max_Model_Order = 7;
noise_var = 150;

T=[];
Tt=[];
Tcv=[];
Tcvs=[];

x = Range*rand(Nos_Samps,1)-Range/2;

f = 5*x.^3  - x.^2 + x;
f_n = 5*x.^3  - x.^2 + x + noise_var*randn(size(x));

xt = (-Range/2:0.01:Range/2)';
tt = 5*xt.^3 - xt.^2 + xt;

[i,j]=sort(x);

X=x.^0;
Xt=xt.^0;

for k=1:Max_Model_Order
    X=[X x.^k];
    Xt=[Xt xt.^k]; 
    
    w_hat = inv(X'*X)*X'*f_n;
    f_hat = X*w_hat;
    f_test = Xt*w_hat;
    
    [cve, cvs] = cross_val(X, f_n);
    T  = [T; mean((f_n - f_hat).^2)];
    Tt = [Tt; mean((tt - f_test).^2)];  
    Tcv = [Tcv; cve];
    Tcvs=[Tcvs;cvs];
    
    plot(i,f(j),'-');
    hold on
    plot(i,f_n(j),'.g')
    plot(i,f_hat(j),'-r')
    hold off
    pause(1)
end

figure
plot(1:Max_Model_Order,T,'dr--');
hold;
plot(1:Max_Model_Order,Tt,'og-');
plot(1:Max_Model_Order,Tcv,'ok-');
%errorbar(1:Max_Model_Order,Tcv,Tcvs,'sk-');

