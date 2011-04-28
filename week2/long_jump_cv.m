clear
Max_Model_Order = 3;
N_years = 20;
T=[];
Tt=[];
Tcv=[];
Tcvs=[];

D=load('long_jump_data.txt');
x=D(1:N_years,1);
t=D(1:N_years,2);

xt=D(N_years+1:end,1);
tt=D(N_years+1:end,2);

X=x.^0;
Xt=xt.^0;

for k=1:Max_Model_Order
    X=[X x.^k];
    Xt=[Xt xt.^k];
    f_hat = X*inv(X'*X)*X'*t;
    ft_hat = Xt*inv(X'*X)*X'*t;
    [cve, cvs] = cross_val(X, t);
    T  = [T; mean((t - f_hat).^2)];
    Tt  = [Tt; mean((tt - ft_hat).^2)];
    Tcv = [Tcv; cve];
    Tcvs = [Tcvs; cvs];
end
subplot(121)
plot(1:Max_Model_Order,T,'rd-');
hold on;
plot(1:Max_Model_Order,Tcv,'gd-');
%errorbar(1:Max_Model_Order,Tcv,Tcvs,'sk-');
subplot(122)
plot(1:Max_Model_Order,Tt,'ro-');