clear
Max_Model_Order = 1;
Range = 20;
N=30;
sigma = 0.25;
xt = [-Range/2:0.1:Range/2]'; 
xt(find(xt == 0))=[];
ft = sin(xt)./xt;
x = Range.*rand(N,1) - Range/2;
f = sin(x)./x;
[i,j] = sort(x);
e = sigma*randn(N,1);
t = f + e;

alpha = 100;
k=0.15;
X=kernel_func(x,x,'gauss',k,k)';
Xt=kernel_func(x,xt,'gauss',k,k)';
pos_cov = sigma*inv(X'*X + (sigma/alpha)*eye(N));
mu = pos_cov*X'*t./sigma;
pred_mean = Xt*mu;
pred_cov = diag(Xt*pos_cov*Xt');

%plot(i,f(j),'-');
hold on
plot(i,t(j),'k.');
plot(xt,ft)
plot(xt,pred_mean,'r')
plot(xt,pred_mean+sqrt(pred_cov),'r');
plot(xt,pred_mean-sqrt(pred_cov),'r');
