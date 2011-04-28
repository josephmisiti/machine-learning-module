clear
N=25;
xx=rand(N,1)-0.5;
xt=[-1:0.01:1]';
Nt=size(xt,1);

L=[];
Lt=[];

X=[xx.^2 xx];
Xt=[xt.^2 xt];

w0=-0.5;
w1=0.5;
sigma=0.05;

f=X*[w1; w0];
ft=Xt*[w1;w0];

t=f+sigma*randn(N,1);

for alpha = 10:-0.1:0.1
    w_hat = inv(X'*X + eye(2)*sigma/alpha)*X'*t;

    likelihood=gauss(t',eye(N)*sigma,w_hat'*X');
    test_likelihood=gauss(ft',eye(Nt)*sigma,w_hat'*Xt');
    L=[L;likelihood];
    Lt=[Lt;test_likelihood];
end
figure
subplot(211)
plot(10:-0.1:0.1,L)
subplot(212)
plot(10:-0.1:0.1,Lt,'r')


