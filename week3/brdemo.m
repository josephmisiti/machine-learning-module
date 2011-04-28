N=25;
xx=rand(N,1)-0.5;
X=[xx.^2 xx];
w0=-0.5;
w1=0.5;
sigma=0.05;
alpha = 1;

f=X*[w1; w0];
t=f+sigma*randn(N,1);
that = X*inv(X'*X + eye(2)*sigma/alpha)*X'*t;

[x,y]=meshgrid(-1:0.05:1,-1:0.05:1);
[n,n]=size(x);
W=[reshape(x,n*n,1) reshape(y,n*n,1)];
mu = inv(X'*X + eye(2)*sigma/alpha)*X'*t;
C = sigma*inv(X'*X + eye(2)*sigma/alpha);
prior=reshape(gauss([0 0],eye(2)*alpha,W),[n n]);
likelihood=reshape(gauss(t',eye(N)*sigma,W*X'),[n n]);
posterior=reshape(gauss(mu',C, W),[n n]);

figure
subplot(2,2,1)
contour(x,y,prior)
hold
plot(w1,w0,'o')
subplot(2,2,2)
contour(x,y,likelihood)
hold
plot(w1,w0,'o')
subplot(2,2,3)
contour(x,y,posterior);
hold
plot(w1,w0,'o')
subplot(2,2,4)
hold
[X_sort,ind_sort]=sort(xx(:,1));
plot(xx(ind_sort,1),f(ind_sort));
plot(xx(ind_sort,1),t(ind_sort),'.');
plot(xx(ind_sort,1),that(ind_sort),'r');
